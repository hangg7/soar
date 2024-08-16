import threading
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import UInt8

from .renderer import Renderer, RenderTask
from .utils import CameraState, LightState, ViewerStats, view_lock


class ViewerServer(viser.ViserServer):
    def __init__(
        self,
        *args,
        render_fn: Callable[[CameraState, tuple[int, int]], UInt8[np.ndarray, "H W 3"]],
        camera_state_extras_fn: Callable[[], dict[str, Any]] = lambda: {},
        stats: Optional[ViewerStats] = None,
        lock: Optional[threading.Lock] = view_lock,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Hack to disable verbose logging.
        self._server._verbose = False

        # This function is called every time we need to render a new image.
        self.render_fn = render_fn
        # This function is called every time we query the camera state. Useful
        # when we are trying to control what to render interacting with GUI or
        # model state etc.
        self.camera_state_extras_fn = camera_state_extras_fn
        self.stats = stats or ViewerStats()
        self.lock = lock
        self.light_state: Optional[LightState] = None
        self.renderers: dict[int, Renderer] = {}

        self.on_client_disconnect(self._disconnect_client)
        self.on_client_connect(self._connect_client)

        # clients = self.get_clients()
        # for client_id in clients:
        #     client_id.camera.wxyz = [0.0, 0.0, 0.0, 1.0]
        #     client_id.camera.position = [0.0, 0.0, 1.0]

        # Public states.
        self.training_state: Literal[
            "preparing", "training", "paused", "completed"
        ] = "training"

        self.player_state: Literal[
            "preparing", "playing", "paused", "completed"
        ] = "paused"

        self.player_fps: int = 15
        # Private states.
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        self._define_guis()

    def _create_lighting_folder(self):
        self._lighting_folder = self.add_gui_folder("Lighting")
        with self._lighting_folder:
            self._light_color_controller = self.add_gui_rgb(
                "Light Color", initial_value=[1.0, 1.0, 1.0]
            )
            self._light_azimuth_controller = self.add_gui_slider(
                "Light Azimuth", min=0, max=360, step=1, initial_value=0
            )
            self._light_elevation_controller = self.add_gui_slider(
                "Light Elevation", min=-90, max=90, step=1, initial_value=0
            )
            self._light_elevation_controller.on_update(self._rerender)
            self._light_azimuth_controller.on_update(self._rerender)
            self._light_color_controller.on_update(self._rerender)
        self.is_lighting_folder_created = True

    def _define_guis(self):
        self._stats_folder = self.add_gui_folder("Stats")
        with self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.add_gui_markdown(self._stats_text_fn())

        self._training_folder = self.add_gui_folder("Training")
        with self._training_folder:
            self._pause_train_button = self.add_gui_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_trainer_state)
            self._resume_train_button = self.add_gui_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_trainer_state)

            self._train_util_slider = self.add_gui_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self._rerender)

        self._player_folder = self.add_gui_folder("Player")
        with self._player_folder:
            self._resume_playing_button = self.add_gui_button("Play")
            self._resume_playing_button.on_click(self._toggle_playing_buttons)
            self._resume_playing_button.on_click(self._toggle_player_state)
            self._pause_playing_button = self.add_gui_button("Pause")
            self._pause_playing_button.visible = False
            self._pause_playing_button.on_click(self._toggle_playing_buttons)
            self._pause_playing_button.on_click(self._toggle_player_state)

            self._player_fps_slider = self.add_gui_slider(
                "Player FPS", min=5, max=60, step=1, initial_value=15
            )
            self._player_fps_slider.on_update(self._rerender)

        self._rendering_folder = self.add_gui_folder("Rendering")
        with self._rendering_folder:
            self._max_img_res_slider = self.add_gui_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._render_selection = self.add_gui_dropdown(
                "Render Selection", ["RGB", "Depth", "Normals", "Depth Normals"]
            )
            self._render_selection.on_update(self._rerender)
            self._max_img_res_slider.on_update(self._rerender)
        self._animation_folder = self.add_gui_folder("Animation")
        with self._animation_folder:
            self._pose_seq_selection = self.add_gui_dropdown(
                "Pose Sequence", ["Training"]
            )
            self._pose_time_slider = self.add_gui_slider(
                "Pose Frame", min=0, max=42, step=1, initial_value=0
            )
            self._pose_seq_selection.on_update(self._rerender)
            self._pose_time_slider.on_update(self._rerender)
        self._editing_folder = self.add_gui_folder("Text Prompt")
        with self._editing_folder:
            self._text_prompt_controller = self.add_gui_text(
                "Text Prompt", initial_value="Hello, World!"
            )
            self._text_prompt_controller.on_update(self._rerender)
        self._create_lighting_folder()

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_trainer_state(self, _):
        if self.training_state == "completed":
            return
        self.training_state = (
            "paused" if self.training_state == "training" else "training"
        )

    def _toggle_playing_buttons(self, _):
        self._pause_playing_button.visible = not self._pause_playing_button.visible
        self._resume_playing_button.visible = not self._resume_playing_button.visible

    def _toggle_player_state(self, _):
        if self.player_state == "completed":
            return
        self.player_state = "paused" if self.player_state == "playing" else "playing"

    def _rerender(self, _):
        # if self._render_selection.value != "RGB":
        #     self._lighting_folder.remove()
        #     self.is_lighting_folder_created = False
        #     # self._light_color_controller.visible = False
        #     # self._light_azimuth_controller.visible = False
        #     # self._light_elevation_controller.visible = False
        # else:
        #     # print(self._lighting_folder)
        #     if not self.is_lighting_folder_created:
        #         self._create_lighting_folder()
        # pass
        # self._light_color_controller.visible = True
        # self._light_azimuth_controller.visible = True
        # self._light_elevation_controller.visible = True
        self.player_fps = self._player_fps_slider.value
        clients = self.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            azimuth = np.deg2rad(self._light_azimuth_controller.value)
            elev = np.deg2rad(self._light_elevation_controller.value)
            self.light_state = LightState(
                direction=np.array(
                    [
                        np.cos(elev) * np.sin(azimuth),
                        np.sin(elev),
                        -np.cos(elev) * np.cos(azimuth),
                    ],
                    dtype=np.float32,
                ),
                color=np.array(self._light_color_controller.value, dtype=np.float32),
            )
            self.renderers[client_id].submit(
                RenderTask("update", camera_state, self.light_state)
            )

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self.renderers[client_id].running = False
        self.renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self.renderers[client_id] = Renderer(server=self, client=client, lock=self.lock)
        self.renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.atomic():
                camera_state = self.get_camera_state(client)
                self.renderers[client_id].submit(
                    RenderTask("move", camera_state, self.light_state)
                )

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
            extras=self.get_pose_frame(),
            mode=self._render_selection.value,
        )

    def get_pose_frame(self):
        seq = self._pose_seq_selection.value
        frame = self._pose_time_slider.value
        return {"pose_seq": seq, "pose_frame": frame}

    def update_player(self) -> float:
        # while time.time() - self._last_move_time < 0.1:
        #     time.sleep(0.05)
        # print('player state:', self.player_state)
        # print('time delta:', time.time() - self._last_move_time)
        # return
        # while time.time() - self._last_move_time < 0.2:
        #     time.sleep(0.15)
        start_time = time.time()
        with self.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self.renderers) == 0:
            return
        self._pose_time_slider.value = (self._pose_time_slider.value + 1) % 42
        clients = self.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self.renderers[client_id].submit(
                RenderTask("update", camera_state, self.light_state)
            )
        with self.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        end_time = time.time()
        return end_time - start_time
        # self._last_move_time = time.time()

    def update(self, step: int, num_train_rays_per_step: int):
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self.renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.training_state == "training" and self._train_util_slider.value != 1:
            assert (
                self.stats.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.stats.num_train_rays_per_sec
            view_s = self.stats.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self.renderers[client_id].submit(
                        RenderTask("update", camera_state, self.light_state)
                    )
                with self.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.training_state = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""
