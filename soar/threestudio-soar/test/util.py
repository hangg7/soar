import numpy as np


def save_obj_mesh(mesh_path, verts, faces=None, color=None):
    file = open(mesh_path, "w")
    for i, v in enumerate(verts):
        if color is None:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        else:
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], color[i][0], color[i][1], color[i][2])
            )
    if faces is not None:
        for f in faces:
            f_plus = f + 1
            file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


# https://github.com/ratcave/wavefront_reader
def read_mtlfile(fname):
    materials = {}
    with open(fname) as f:
        lines = f.read().splitlines()

    for line in lines:
        if line:
            split_line = line.strip().split(" ", 1)
            if len(split_line) < 2:
                continue

            prefix, data = split_line[0], split_line[1]
            if "newmtl" in prefix:
                material = {}
                materials[data] = material
            elif materials:
                if data:
                    split_data = data.strip().split(" ")

                    # assume texture maps are in the same level
                    # WARNING: do not include space in your filename!!
                    if "map" in prefix:
                        material[prefix] = split_data[-1].split("\\")[-1]
                    elif len(split_data) > 1:
                        material[prefix] = tuple(float(d) for d in split_data)
                    else:
                        try:
                            material[prefix] = int(data)
                        except ValueError:
                            material[prefix] = float(data)

    return materials


def load_obj_mesh_mtl(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    # face per material
    face_data_mat = {}
    face_norm_data_mat = {}
    face_uv_data_mat = {}

    # current material name
    mtl_data = None
    cur_mat = None

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == "v":
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == "vn":
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == "vt":
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == "mtllib":
            mtl_data = read_mtlfile(
                mesh_file.replace(mesh_file.split("/")[-1], values[1])
            )
        elif values[0] == "usemtl":
            cur_mat = values[1]
        elif values[0] == "f":
            # local triangle data
            l_face_data = []
            l_face_uv_data = []
            l_face_norm_data = []

            # quad mesh
            if len(values) > 4:
                f = list(
                    map(
                        lambda x: (
                            int(x.split("/")[0])
                            if int(x.split("/")[0]) < 0
                            else int(x.split("/")[0]) - 1
                        ),
                        values[1:4],
                    )
                )
                l_face_data.append(f)
                f = list(
                    map(
                        lambda x: (
                            int(x.split("/")[0])
                            if int(x.split("/")[0]) < 0
                            else int(x.split("/")[0]) - 1
                        ),
                        [values[3], values[4], values[1]],
                    )
                )
                l_face_data.append(f)
            # tri mesh
            else:
                f = list(
                    map(
                        lambda x: (
                            int(x.split("/")[0])
                            if int(x.split("/")[0]) < 0
                            else int(x.split("/")[0]) - 1
                        ),
                        values[1:4],
                    )
                )
                l_face_data.append(f)
            # deal with texture
            if len(values[1].split("/")) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[1])
                                if int(x.split("/")[1]) < 0
                                else int(x.split("/")[1]) - 1
                            ),
                            values[1:4],
                        )
                    )
                    l_face_uv_data.append(f)
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[1])
                                if int(x.split("/")[1]) < 0
                                else int(x.split("/")[1]) - 1
                            ),
                            [values[3], values[4], values[1]],
                        )
                    )
                    l_face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[1]) != 0:
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[1])
                                if int(x.split("/")[1]) < 0
                                else int(x.split("/")[1]) - 1
                            ),
                            values[1:4],
                        )
                    )
                    l_face_uv_data.append(f)
            # deal with normal
            if len(values[1].split("/")) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[2])
                                if int(x.split("/")[2]) < 0
                                else int(x.split("/")[2]) - 1
                            ),
                            values[1:4],
                        )
                    )
                    l_face_norm_data.append(f)
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[2])
                                if int(x.split("/")[2]) < 0
                                else int(x.split("/")[2]) - 1
                            ),
                            [values[3], values[4], values[1]],
                        )
                    )
                    l_face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[2]) != 0:
                    f = list(
                        map(
                            lambda x: (
                                int(x.split("/")[2])
                                if int(x.split("/")[2]) < 0
                                else int(x.split("/")[2]) - 1
                            ),
                            values[1:4],
                        )
                    )
                    l_face_norm_data.append(f)

            face_data += l_face_data
            face_uv_data += l_face_uv_data
            face_norm_data += l_face_norm_data

            if cur_mat is not None:
                if cur_mat not in face_data_mat.keys():
                    face_data_mat[cur_mat] = []
                if cur_mat not in face_uv_data_mat.keys():
                    face_uv_data_mat[cur_mat] = []
                if cur_mat not in face_norm_data_mat.keys():
                    face_norm_data_mat[cur_mat] = []
                face_data_mat[cur_mat] += l_face_data
                face_uv_data_mat[cur_mat] += l_face_uv_data
                face_norm_data_mat[cur_mat] += l_face_norm_data

    vertices = np.array(vertex_data)
    faces = np.array(face_data)

    norms = np.array(norm_data)
    norms = normalize_v3(norms)
    face_normals = np.array(face_norm_data)

    uvs = np.array(uv_data)
    face_uvs = np.array(face_uv_data)

    out_tuple = (vertices, faces, norms, face_normals, uvs, face_uvs)

    if cur_mat is not None and mtl_data is not None:
        for key in face_data_mat:
            face_data_mat[key] = np.array(face_data_mat[key])
            face_uv_data_mat[key] = np.array(face_uv_data_mat[key])
            face_norm_data_mat[key] = np.array(face_norm_data_mat[key])

        out_tuple += (face_data_mat, face_norm_data_mat, face_uv_data_mat, mtl_data)

    return out_tuple


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == "v":
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == "vn":
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == "vt":
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == "f":
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(
                        lambda x: int(x.split("/")[0]),
                        [values[3], values[4], values[1]],
                    )
                )
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split("/")) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[1]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[1]) != 0:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split("/")) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[2]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[2]) != 0:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


# compute tangent and bitangent
def compute_tangent(vertices, faces, normals, uvs, faceuvs):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)

    # NOTE: traditional version is below

    # pts_tris = vertices[faces]
    # uv_tris = uvs[faceuvs]

    # W = np.stack([pts_tris[::, 1] - pts_tris[::, 0], pts_tris[::, 2] - pts_tris[::, 0]],2)
    # UV = np.stack([uv_tris[::, 1] - uv_tris[::, 0], uv_tris[::, 2] - uv_tris[::, 0]], 1)

    # for i in range(W.shape[0]):
    #     W[i,::] = W[i,::].dot(np.linalg.inv(UV[i,::]))

    # tan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # tan[faces[:,0]] += W[:,:,0]
    # tan[faces[:,1]] += W[:,:,0]
    # tan[faces[:,2]] += W[:,:,0]

    # btan = np.zeros(vertices.shape, dtype=vertices.dtype)
    # btan[faces[:,0]] += W[:,:,1]
    # btan[faces[:,1]] += W[:,:,1]
    # btan[faces[:,2]] += W[:,:,1]

    # normalize_v3(tan)

    # ndott = np.sum(normals*tan, 1, keepdims=True)
    # tan = tan - ndott * normals

    # normalize_v3(btan)
    # normalize_v3(tan)

    # tan[np.sum(np.cross(normals, tan) * btan, 1) < 0,:] *= -1.0

    return tan, btan

import torch
import math

# gaussian splatting functions
def convert_pose(C2W):
    flip_yz = torch.eye(4, device=C2W.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def get_projection_matrix_gaussian(
    znear, zfar, fovX, fovY, device="cuda", cxcy=None, img_wh=None, z_sign=1.0
):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    if cxcy is not None and img_wh is not None:
        cx, cy = cxcy
        W, H = img_wh
        P[0, 2] = (2.0 * cx - W) / W
        P[1, 2] = (2.0 * cy - H) / H
    else:
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
    # P[2] *= -1
    return P


def get_fov_gaussian(P):
    tanHalfFovX = 1 / P[0, 0]
    tanHalfFovY = 1 / P[1, 1]
    fovY = math.atan(tanHalfFovY) * 2
    fovX = math.atan(tanHalfFovX) * 2
    return fovX, fovY


def get_cam_info_gaussian(
    c2w, fovx, fovy, znear, zfar, cxcy=None, img_wh=None, back=False
):
    c2w_converted = convert_pose(c2w)
    world_view_transform = torch.inverse(c2w_converted)

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()

    projection_matrix = get_projection_matrix_gaussian(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
        cxcy=cxcy,
        img_wh=img_wh,
    )
    if back:
        projection_matrix[2] *= -1
    projection_matrix = projection_matrix.transpose(0, 1).cuda()

    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center


if __name__ == "__main__":
    pts, tri, nml, trin, uvs, triuv = load_obj_mesh(
        "/home/ICT2000/ssaito/Documents/Body/tmp/Baseball_Pitching/0012.obj", True, True
    )
    compute_tangent(pts, tri, uvs, triuv)
