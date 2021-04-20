"""
Code for creating the tfRecords containing the whol eshapenet datset
"""

import os
import sys
sys.path.append('/orion/u/ianhuang/pyRender/lib')

from tqdm import tqdm
import trimesh
import objloader
import numpy as np
import render
import skimage.io as sio
from skimage import img_as_ubyte
from random import shuffle

# DATADIR = "/orion/group/ShapeNetCore.v2/"
DATADIR = "/orion/group/ShapeNetManifold_10000"

info = {'Height': 224, 'Width': 224, 'fx':575, 'fy':575, 'cx': 111.5, 'cy': 111.5}
render.setup(info)

# TARGETDIR = "/orion/u/ianhuang/cvxstyle_tfrecords/"
# TARGETDIR = "/orion/u/ianhuang/cvxstyle_tfrecords_v3/"
TARGETDIR = "/tmp/cvxstyle_tfrecords_debug/"

# CHECKDIR = "/orion/u/ianhuang/cvxstyle_depth_v4/"


synset_classes = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}



import tensorflow as tf

if not os.path.exists(TARGETDIR):
    os.makedirs(TARGETDIR)

#if CHECKDIR is not None:
#    if not os.path.exists(CHECKDIR):
#        os.makedirs(CHECKDIR)

extra_small = False

if __name__=='__main__':

    num_surface_samples = 1024
    num_bbox_samples = 1024

    num_views = 24

    synsets = [el for el in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, el))]
    synset_paths = [os.path.join(DATADIR, el) for el in synsets]

    for class_idx, synset_path in enumerate(synset_paths):

        # make train/test split
        # make ONE tf.record each.
        # surf_samples, bbox_samples and depth need to be flattened first before
        # stored into tf.record

        model_ids = [el for el in os.listdir(synset_path) if os.path.isdir(os.path.join(synset_path, el))]
        model_paths = [os.path.join(synset_path, el) for el in model_ids]

        if extra_small:
            print("EXTRA SMALL DATASET")
            model_paths = model_paths[:20]
            model_ids = model_ids[:20]

        shuffle(model_paths)
        status = ['train']*int(0.8 * len(model_paths))
        status += ['test']*(len(model_paths) - len(status))

        assert len(status) == len(model_paths)

        synset_id = os.path.basename(synset_path)
        # opening the writer
        f_train_record = tf.io.TFRecordWriter(os.path.join(TARGETDIR,
                                                           "{}-{}-data.tfrecords".
                                                           format(synset_classes[synset_id], "train")))
        f_test_record = tf.io.TFRecordWriter(os.path.join(TARGETDIR,
                                                          "{}-{}-data.tfrecords".
                                                          format(synset_classes[synset_id], "test")))

        for model_idx, model_path in enumerate(model_paths):
            obj_dir = os.path.join(model_path, 'models')
            img_dir = os.path.join(model_path, 'images') # this is just the texture

            if not os.path.exists(obj_dir):
                print("{} does not exist. Moving onto next model".format(obj_dir))
                continue

            obj_path = os.path.join(obj_dir, 'model_normalized.obj')

            # load obj, get vertex and faces
            V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadSimpleOBJ_manifold(obj_path) # LoadSimpleOBJ(obj_path)
            # with open(obj_path, 'r') as f:
            #     obj_dict = trimesh.exchange.obj.load_obj(f)
            context = render.SetMesh(V, F) # plop the shape into the render

            mesh = trimesh.Trimesh(vertices=V, faces=F) # face_normals=FN)
            proximityquery = trimesh.proximity.ProximityQuery(mesh)

            # surfaces sampling: assuming trimesh subdivision is even enough.
            surf_coord, _ = trimesh.sample.sample_surface(mesh, num_surface_samples)
            surf_distance = proximityquery.signed_distance(surf_coord)
            max_surf_distance = np.max(np.abs(surf_distance))
            surf_samples = np.hstack((surf_coord, surf_distance.reshape(-1, 1)))

            # bounding box sampling: assuming bounding box tightly fits the vertex
            bounding_box_min, bounding_box_max = np.min(V, axis=0), np.max(V, axis=0)
            x_samples = np.random.uniform(bounding_box_min[0], bounding_box_max[0], num_bbox_samples)
            y_samples = np.random.uniform(bounding_box_min[1], bounding_box_max[1], num_bbox_samples)
            z_samples = np.random.uniform(bounding_box_min[2], bounding_box_max[2], num_bbox_samples)
            bbox_coord = np.vstack((x_samples, y_samples, z_samples)).transpose()
            signed_bbox_distances = proximityquery.signed_distance(bbox_coord)
            bbox_samples = np.hstack((bbox_coord,
                                      signed_bbox_distances.reshape(-1, 1)))

            # check how these distances compare to surf_distance
            max_norm = np.max(np.linalg.norm(V,axis=1))

            # for depth
            # how many depth views?
            # 1024 -> 32 samples around, 32 samples up and down
            thetas = np.linspace(0, np.pi, 5+2)
            thetas = thetas[1:-1] # cutting off the ends
            phis = np.linspace(0, 2*np.pi, 4)
            r = 6 * max_norm
            # sampling points on a sphere around the object
            depth_views = []

            print('surface and bbox sampling done for model {}/{} in class {}/{}'\
                  .format(model_idx+1,
                          len(model_paths),
                          class_idx+1,
                          len(synset_paths)))
            print('Generating depth views for model {}/{} in class {}/{}'\
                  .format(model_idx+1,
                          len(model_paths),
                          class_idx+1,
                          len(synset_paths)))

            for theta in thetas:
                for phi in phis:
                    # make camera transformation matrix
                    center_x = r*np.cos(phi)*np.sin(theta)
                    center_y = r*np.sin(phi)*np.sin(theta)
                    center_z = r*np.cos(theta)

                    to_origin = np.array([center_x, center_y, center_z])
                    norm =  np.linalg.norm(to_origin)
                    unit_to_origin = to_origin/norm

                    camera2world = np.identity(4)
                    camera2world[:3, 3] = to_origin

                    # fill in the rotation part of the matrix
                    forward = - unit_to_origin
                    right0 = forward[1]/(np.sqrt(forward[0]**2 + forward[1]**2))
                    right1 = np.sqrt(1 - forward[1]**2 / (forward[0]**2 + forward[1]**2))
                    right2 = 0
                    right = np.array([right0, right1, right2])
                    # figure out if you need to flip right
                    if center_x >= 0 and right1 < 0: # you need to flip this
                        right = -right
                    up = np.cross(right, forward) # guaranteed to be unit normal

                    camera2world[:3, 0] = right
                    camera2world[:3, 1] = up
                    camera2world[:3, 2] = forward
                    world2camera = np.linalg.inv(camera2world).astype('float32')

                    # rendering
                    render.render(context, world2camera)
                    depth = render.getDepth(info)
                    depth = depth / np.max(depth)

                    # saving depth
                    # TODO: save it for verfication purposes
                    # save at CHECKDIR/{class}/{train or validation}/{model_id}/
                    # depth_save_dir = os.path.join(CHECKDIR, synset_classes[synset_id])
                    # depth_save_dir = os.path.join(depth_save_dir, status[model_idx])
                    # depth_save_dir = os.path.join(depth_save_dir,
                    #                               os.path.basename(model_path))
                    # if not os.path.exists(depth_save_dir):
                    #     os.makedirs(depth_save_dir)
                    # sio.imsave(os.path.join(
                    #     depth_save_dir,
                    #     'depth_t{}_p{}.png'.format(theta, phi)),
                    #            img_as_ubyte(depth))
                    depth_views.append(depth)



            # NOTE: Assume that RGB information is irrelevant for our purposes.
            rgb = np.zeros((num_views, 137, 137, 3)).astype(np.float32) # views, h, w, d

            depth_views = np.array(depth_views)

            classname = synset_classes[os.path.basename(os.path.dirname(model_path))]
            objname = os.path.basename(model_path)
            name = "{}-{}".format(classname, objname)

            import ipdb; ipdb.set_trace()
            example = tf.train.Example(features =
                                       tf.train.Features(feature =
                                                         {'rgb': tf.train.Feature(float_list=tf.train.FloatList(value=rgb.reshape(-1))),
                                                          'depth': tf.train.Feature(float_list=tf.train.FloatList(value=depth_views.reshape(-1))),
                                                          'bbox_samples': tf.train.Feature(float_list=tf.train.FloatList(value=bbox_samples.reshape(-1))),
                                                          'surf_samples': tf.train.Feature(float_list=tf.train.FloatList(value=surf_samples.reshape(-1))),
                                                          'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(name,
                                                                                                 encoding='utf-8')]))
                                                         }
                                       ))

            example = example.SerializeToString()

            if status[model_idx] == 'train':
                # write
                f_train_record.write(example)

            elif status[model_idx] == 'test':
                # write
                f_test_record.write(example)

            else:
                raise ValueError("invalid status for sample {}".\
                                 format(status[model_idx]))


        # closing files
        f_train_record.close()
        f_test_record.close()

    render.Clear()

