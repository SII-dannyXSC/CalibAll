import numpy as np

ROBOMIND_SCENE1_SET = (
    "241021_close_trash_bin_1",
    "241021_insert_marker_1",
    "241021_open_trash_bin_1",
    "241021_remove_marker_1",
    "241022_lamp_off_1",
    "241022_lamp_on_1",
    "241022_side_pull_close_drawer_1",
    "241022_side_pull_open_drawer_1",
    "241023_pick_pear_from_bowl_1",
    "241023_pick_pear_from_bowl_2",
    "241023_place_pear_in_bowl_1",
    "241023_place_pear_in_bowl_2",
    "241223_upright_cup",
    "apples_placed_on_a_ceramic_plate",
    "bananas_placed_on_a_ceramic_plate",
    "blue_cub_on_pink",
    "bread_is_placed_on_a_ceramic_plate",
    "cap_close_dustbin",
    "chili_peppers_are_placed_on_a_ceramic_plate",
    "chili_peppers_are_placed_on_the_right_side_of_a_plastic_tray",
    "close_cap_lid",
    "close_cap_trash_can",
    "close_garbage_bin",
    "close_lid",
    "cucumber_placed_on_a_ceramic_plate",
    "flip_marco_cup",
    "mobile_marco_cup",
    "open_cap_lid",
    "open_cap_trash_can_1",
    "open_lid",
    "open_the_drawer",
    "pick_plate_from_plate_rack",
    "pick_up_strawberry_from_bowl",
    "pick_up_strawberry_in_bowl",
    "piled_on_stack_blue_block_on_pink_block",
    "place_in_block_in_plate_1",
    "place_in_block_on_table",
    "place_in_bread_in_plate",
    "place_in_bread_on_plate_1",
    "place_in_bread_on_plate_2",
    "place_in_bread_on_table",
    "place_in_bread_on_table_2",
    "place_in_pick_up_and_throw_away_1",
    "place_in_toy",
    "place_marker",
    "place_plate_in_plate_rack",
    "put_potatoes_on_a_ceramic_plate",
    "put_the_bread_on_the_table",
    "put_the_cucumber_on_the_left_side_of_the_bowl",
    "put_the_red_apple_in_the_bowl",
    "put_the_steamed_buns_on_a_ceramic_plate",
    "red_apple_placed_in_the_center_of_the_desktop",
    "slide_close_drawer",
    "slide_close_drawer_1_1",
    "slide_open_drawer",
    "slide_open_drawer_1",
    "stick_target_blue_on_the_pink_obejct",
    "strawberries_on_a_ceramic_plate",
    "take_marker_pen",
    "turn_off_lamp",
    "turn_on_desk_lamp",
    "side_pull_drawer",
    "side_opening_drawer",
    "yellow_bananas_placed_on_a_plastic_tray",
    "yellow_square_placed_on_ceramic_plate",
)

ROBOMIND_SCENE2_SET = (
    "piled_on_yellow_block_on_purple_block",
    "place_in_bread_in_basket_1",
    "place_in_bread_in_bread_machine",
    "place_in_bread_on_table_1",
    "place_in_fruit_in_fruit_basket",
    "place_in_purple_block_in_plate",
    "place_in_purple_block_on_table",
    "place_in_take_bread_and_put_in_plate",
    "place_in_yellow_block_on_table",
    "twist_knob_start_bread_machine",
)

ROBOMIND_SCENE3_SET = (
    "close_cap_tool_box",
    "close_cap_trash_can_1",
    "close_cap_trash_can_2",
    "close_drawer",
    "close_trash",
    "open_cap_tool_box",
    "open_cap_trash_can",
    "open_cap_trash_can_2",
    "open_drawer",
    "open_trash",
    "pick_apple_into_chest",
    "pick_bread_desk",
    "pick_bread_into_plate",
    "pick_drawer_tool",
    "place_in_block_1",
    "place_in_block_tennis_ball",
    "place_in_cylinder",
    "place_in_fruit",
    "place_in_fruit_bread",
    "place_in_fruit_in_basket",
    "place_in_fruit_on_table",
    "place_in_rectangular_prism",
    "place_in_shape",
    "place_in_trash",
    "place_trash",
    "pull_across_pull_in_basket",
    "push_across_push_away_basket",
    "rotate_close_cabinet",
    "rotate_open_cabinet",
    "slide_close_cabinet",
    "slide_open_cabinet",
)

ROBOMIND_SCENE4_SET = (
    "place_in_bread_in_basket",
    "place_in_pick_up_and_throw_away",
    "slide_close_drawer_1",
)

ROBOMIND_UR5E_SET1 = (
    'bread_in_basket_1'
)

ROBOMIND_UR5E_SET2 = (
    "triangle_bread_in_basket",
    'red_pepper_on_table',
    'red_pepper_in_basket',
    'triangle_bread_in_basket_1',
    'triangle_bread_on_table',
    'yellow_pepper_in_basket',
    'yellow_pepper_in_basket_1',
    'yellow_pepper_on_table',
    'red_pepper_in_basket',
    'put_egg_in_pot',
    'pick_up_egg',
    'green_pepper_on_table',
    'green_pepper_in_basket_1',
    'green_pepper_in_basket',
    'bread_on_table',
    'bread_in_basket_old'
)

# label_out/robomind_ur/recalib_review.json 中 decision == "recalibrate" 的任务
# （键为 b1_x/<task_dir>，此处仅 LeRobot 任务目录名，与 get_intrinsic_extrinsic 的 task_name 一致）
ROBOMIND_UR5E_SET3 = (
    "green_pepper_on_plate",
    "green_pepper_on_the_table",
    "pick_up_green_pepper_from_plate",
    "pick_up_green_pepper_from_table",
    "pick_up_red_pepper_from_plate",
    "pick_up_red_pepper_from_table",
    "pick_up_yellow_pepper_from_plate",
    "pick_up_yellow_pepper_from_plate_copy_1734079773826",
    "pick_up_yellow_pepper_from_table",
    "pick_up_yellow_pepper_from_table_copy_1734079574938",
    "pickupthebananafromtheplate",
    "pickupthebananafromthetable",
    "placebananaonaplate",
    "putthebananaonthetable",
    "red_pepper_in_table",
    "red_pepper_on_plate",
    "yellow_pepper_on_plate",
    "yellow_pepper_on_plate_copy_1734079761061",
)


# -----------------------------------------------------------------------------
# OXE / LeRobot：各数据集根目录 intrinsic.txt 与 tsfm.txt（不含 episode 子目录）。
# 键名与 convert/oxe/lerobot_output 下子目录名一致。
# 单相机：顶层为 INTRINSIC / EXTRINSIC（list，可与 np.array 互转）。
# 多相机：顶层为视角名，其下为 INTRINSIC / EXTRINSIC。
# 值为 None 表示暂无标定，需自行补全。
# 旧名对照（仅供参考）：
#   berkeley_ur5e -> berkeley_autolab_ur5
#   buds -> austin_buds_dataset_converted_externally_to_rlds
#   sailor -> austin_sailor_dataset_converted_externally_to_rlds
#   non_prehen -> kaist_nonprehensile_converted_externally_to_rlds
#   qut_dext -> qut_dexterous_manpulation
#   ucsd_kitchen -> ucsd_kitchen_dataset_converted_externally_to_rlds
#   utokyo_xarm7 -> utokyo_xarm_pick_and_place_converted_externally_to_rlds
#   nyu_franka_play_image* -> nyu_franka_play_dataset_converted_externally_to_rlds 下子键
# -----------------------------------------------------------------------------

OXE_LEROBOT_DATASETS = {
    "asu_table_top_converted_externally_to_rlds": None,
    "austin_buds_dataset_converted_externally_to_rlds": {
        "INTRINSIC": [
            [211.118, 0, 64],
            [0, 211.118, 64],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [-0.0038, 0.994, -0.109, -0.0409],
            [0.8874, -0.0469, -0.4585, -0.484],
            [-0.4609, -0.0985, -0.882, 1.428],
            [0, 0, 0, 1],
        ],
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "INTRINSIC": [
            [145.347, 0, 64],
            [0, 145.347, 64],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.9213, 0.3859, -0.0486, -0.4771],
            [0.3054, -0.7952, -0.5239, -0.313],
            [-0.2408, 0.4678, -0.8504, 0.8538],
            [0, 0, 0, 1],
        ],
    },
    "berkeley_autolab_ur5": {
        "INTRINSIC": [
            [410.698, 0, 320],
            [0, 410.698, 240],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.6062, 0.7953, 0.0048, 0.3449],
            [0.4875, -0.3668, -0.7923, 0.1993],
            [-0.6284, 0.4826, -0.6101, 0.4464],
            [0, 0, 0, 1],
        ],
    },
    "berkeley_fanuc_manipulation": None,
    "cmu_play_fusion": None,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": None,
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "INTRINSIC": [
            [523.096, 0, 320],
            [0, 523.096, 240],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.6252, 0.7761, 0.0824, -0.3496],
            [0.5018, -0.3189, -0.804, -0.3243],
            [-0.5978, 0.544, -0.5888, 0.8719],
            [0, 0, 0, 1],
        ],
    },
    "kuka": None,
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "observation.images.image": {
            "INTRINSIC": [
                [200.859, 0, 64],
                [0, 200.859, 64],
                [0, 0, 1],
            ],
            "EXTRINSIC": [
                [0.5352, -0.8447, 0.0072, -0.1566],
                [-0.2742, -0.1818, -0.9443, 0.6264],
                [0.799, 0.5035, -0.3289, 0.9773],
                [0, 0, 0, 1],
            ],
        },
        "observation.images.image_additional_view": {
            "INTRINSIC": [
                [171.735, 0, 64],
                [0, 171.735, 64],
                [0, 0, 1],
            ],
            "EXTRINSIC": [
                [-0.792, -0.6066, 0.0687, 0.6077],
                [-0.3269, 0.3263, -0.8869, 0.3333],
                [0.5156, -0.7249, -0.4568, 1.5102],
                [0, 0, 0, 1],
            ],
        },
    },
    "qut_dexterous_manpulation": {
        "INTRINSIC": [
            [379.781, 0, 320],
            [0, 379.781, 240],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.8908, 0.4531, -0.0339, -0.4808],
            [0.1973, -0.453, -0.8694, 0.0158],
            [-0.4093, 0.7678, -0.4929, 1.4729],
            [0, 0, 0, 1],
        ],
    },
    "robo_set": None,
    "toto": {
        "INTRINSIC": [
            [569.537, 0, 320],
            [0, 569.537, 240],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.8986, 0.4315, 0.0795, -0.3944],
            [0.2157, -0.2767, -0.9364, 0.0679],
            [-0.3821, 0.8586, -0.3417, 1.1462],
            [0, 0, 0, 1],
        ],
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "INTRINSIC": [
            [480.907, 0, 320],
            [0, 480.907, 240],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.9965, 0.0782, 0.03, -0.373],
            [0.0349, -0.062, -0.9975, 0.2569],
            [-0.0761, 0.995, -0.0645, 0.7366],
            [0, 0, 0, 1],
        ],
    },
    "ut": None,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "INTRINSIC": [
            [219.296, 0, 112],
            [0, 219.296, 112],
            [0, 0, 1],
        ],
        "EXTRINSIC": [
            [0.1472, 0.9887, -0.0292, -0.0768],
            [0.5752, -0.1096, -0.8107, 0.0077],
            [-0.8047, 0.1025, -0.5848, 1.0566],
            [0, 0, 0, 1],
        ],
    },
}


def _get_oxe_lerobot_arrays(dataset_name, camera_name):
    """从 OXE_LEROBOT_DATASETS 解析内参 / 外参，返回 float64 ndarray。

    单相机：dict 顶层含 INTRINSIC / EXTRINSIC，camera_name 可忽略（仍建议传 "image" 等）。
    多相机：顶层为视角名子 dict，须用 camera_name 选中（如 image、image_additional_view）。
    """
    cfg = OXE_LEROBOT_DATASETS.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"OXE LeRobot 数据集 {dataset_name!r} 在 OXE_LEROBOT_DATASETS 中标定为 None，暂无可用矩阵。"
        )
    if "INTRINSIC" in cfg and "EXTRINSIC" in cfg:
        return np.asarray(cfg["INTRINSIC"], dtype=np.float64), np.asarray(
            cfg["EXTRINSIC"], dtype=np.float64
        )
    # 多视角：子键 -> {INTRINSIC, EXTRINSIC}
    if camera_name in cfg:
        v = cfg[camera_name]
        return np.asarray(v["INTRINSIC"], dtype=np.float64), np.asarray(
            v["EXTRINSIC"], dtype=np.float64
        )
    if camera_name in (None, "", "default", "main") and "image" in cfg:
        v = cfg["image"]
        return np.asarray(v["INTRINSIC"], dtype=np.float64), np.asarray(
            v["EXTRINSIC"], dtype=np.float64
        )
    valid = list(cfg.keys())
    raise ValueError(
        f"OXE LeRobot 数据集 {dataset_name!r} 为多视角，需要有效的 camera_name；"
        f"收到 {camera_name!r}，可选: {valid}"
    )


def get_intrinsic_extrinsic(dataset_name, task_name, camera_name, episode):
    INTRINSIC, EXTRINSIC = None, None
    if dataset_name in OXE_LEROBOT_DATASETS:
        INTRINSIC, EXTRINSIC = _get_oxe_lerobot_arrays(dataset_name, camera_name)
    # elif dataset_name == "rdt":
    #     if camera_name == "observation.images.camera_top":
    #         INTRINSIC = np.array([
    #             [626.78735,   0.,      320.     ],
    #             [  0.,      626.78735, 240.     ],
    #             [  0.,        0.,        1.     ]
    #         ])
    #         EXTRINSIC = np.array([
    #             [ 0.9466,  0.2998,  0.1184, -0.0695],
    #             [ 0.3150, -0.7824, -0.5372, -0.4109],
    #             [-0.0684,  0.5459, -0.8351,  1.4191],
    #             [ 0.0000,  0.0000,  0.0000,  1.0000]
    #         ])
    elif dataset_name == "robomind/agilex_3rgb" or dataset_name == 'rdt':
        INTRINSIC = np.array([
            [466.6021 ,   0.     , 320.     ],
            [  0.     , 466.60208, 240.     ],
            [  0.     ,   0.     ,   1.     ],
        ])
        EXTRINSIC = np.array([
            [ 0.0548, -0.9980, -0.0303,  0.0049],
            [ 0.0488,  0.0329, -0.9983,  0.8536],
            [ 0.9973,  0.0532,  0.0505, -0.2315],
            [ 0.0000,  0.0000,  0.0000,  1.0000]
        ])
    elif dataset_name == "robomind/ur_1rgb":
        if task_name in ROBOMIND_UR5E_SET1:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [626.0025,   0.0, 320.0],
                    [  0.0,   626.0025, 240.0],
                    [  0.0,     0.0,    1.0]
                ])
                EXTRINSIC = np.array([
                    [ 0.8126,  0.5799, -0.0573,  0.1418],
                    [ 0.3639, -0.5818, -0.7274, -0.1358],
                    [-0.4552,  0.5702, -0.6839,  1.5654],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        elif task_name in ROBOMIND_UR5E_SET2:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [637.6551  , 0.   ,   320.    ],
                    [  0.   ,  637.6551, 240.    ],
                    [  0.    ,   0.,       1.    ]
                ])
                EXTRINSIC = np.array([
                    [ 0.7364,  0.6470, -0.1978,  0.4419],
                    [ 0.3284, -0.5973, -0.7317, -0.1494],
                    [-0.5915,  0.4739, -0.6523,  1.5198],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        elif task_name in ROBOMIND_UR5E_SET3:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [626.78735,   0.,      320.     ],
                    [  0.,      626.78735, 240.     ],
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [ 0.9466,  0.2998,  0.1184, -0.0695],
                    [ 0.3150, -0.7824, -0.5372, -0.4109],
                    [-0.0684,  0.5459, -0.8351,  1.4191],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        else:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [611.42444,   0.,      320.     ],
                    [  0.,      611.42444, 240.     ],
                    [  0.     ,   0.    ,    1.     ]
                ])
                EXTRINSIC = np.array([
                    [ 0.8856,  0.4642, -0.0155, -0.0623],
                    [ 0.4035, -0.7853, -0.4696, -0.3981],
                    [-0.2302,  0.4096, -0.8828,  1.3294],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
    elif dataset_name == "robomind/franka_3rgb":
        if task_name in ROBOMIND_SCENE1_SET:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [895.93896,    0.0,   640.0],
                    [   0.0,   895.9389,  360.0],
                    [   0.0,      0.0,     1.0]
                ])
                EXTRINSIC = np.array([
                    [-0.0124,  0.9987,  0.0495, -0.1512],
                    [ 0.9133,  0.0315, -0.4061, -0.5770],
                    [-0.4071,  0.0402, -0.9125,  1.2903],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_right":
                INTRINSIC = np.array([
                    [572.993,    0.0, 320.0],
                    [  0.0,   572.993, 240.0],
                    [  0.0,     0.0,    1.0]
                ])
                EXTRINSIC = np.array([
                    [ 0.5394, -0.8300, -0.1421, -0.4130],
                    [-0.5207, -0.1961, -0.8309,  0.6093],
                    [ 0.6618,  0.5222, -0.5379,  0.7001],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_left":
                INTRINSIC = np.array([
                    [589.6532,  0.0,   320.0],
                    [  0.0,  589.6532, 240.0],
                    [  0.0,    0.0,     1.0]
                ])
                EXTRINSIC = np.array([
                    [-0.8502, -0.5217, -0.0703,  0.7919],
                    [-0.2253,  0.4813, -0.8471,  0.2301],
                    [ 0.4757, -0.7044, -0.5268,  0.9254],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        elif task_name in ROBOMIND_SCENE2_SET:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [813.46466,   0.,      640.     ],
                    [  0.,      813.46466, 360.     ],
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [ 0.0664,  0.9978,  0.0071, -0.1501],
                    [ 0.8719, -0.0546, -0.4866, -0.5044],
                    [-0.4852,  0.0386, -0.8736,  1.3062],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_left":
                INTRINSIC = np.array([
                    [565.8554,   0.,     320.    ],
                    [  0.,     565.8554, 240.    ],
                    [  0.,       0.,       1.    ]
                ])
                EXTRINSIC = np.array([
                    [-0.8823, -0.4702, -0.0181,  0.5778],
                    [-0.1968,  0.4038, -0.8934,  0.4902],
                    [ 0.4275, -0.7847, -0.4488,  0.8107],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_right":
                INTRINSIC = np.array([
                    [582.68304,   0.,      320.     ],
                    [  0.,      582.68304, 240.     ],
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [ 0.7015, -0.6757, -0.2266, -0.4034],
                    [-0.5197, -0.2674, -0.8114,  0.7152],
                    [ 0.4877,  0.6870, -0.5387,  0.7601],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        elif task_name in ROBOMIND_SCENE3_SET:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [601.61096,   0.,      320.     ],
                    [  0.,      601.6109,  240.     ],  
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [-0.0471,  0.9989, -0.0033, -0.0831],
                    [ 0.9112,  0.0416, -0.4098, -0.5489],
                    [-0.4092, -0.0223, -0.9122,  1.3287],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_left":
                INTRINSIC = np.array([
                    [562.6102,   0.,      320.    ],
                    [  0.,     562.6102, 240.    ],
                    [  0.,       0.,       1.    ]
                ])
                EXTRINSIC = np.array([
                    [-0.8383, -0.5380, -0.0882,  0.5945],
                    [-0.2175,  0.4783, -0.8509,  0.4496],
                    [ 0.5000, -0.6941, -0.5179,  0.8438],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_right":
                INTRINSIC = np.array([
                    [563.9307,   0.,      320.    ],
                    [  0.,     563.9307, 240.    ],
                    [  0.,       0.,       1.    ]
                ])
                EXTRINSIC = np.array([
                    [ 0.6570, -0.7288, -0.1929, -0.3486],
                    [-0.5179, -0.2504, -0.8180,  0.7135],
                    [ 0.5479,  0.6373, -0.5419,  0.6929],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
        elif task_name in ROBOMIND_SCENE4_SET:
            if camera_name == "observation.images.camera_top":
                INTRINSIC = np.array([
                    [821.29034,   0.,      640.     ],
                    [  0.,      821.29034, 360.     ],
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [-8.5596e-03,  9.9977e-01, -1.9834e-02, -8.3733e-02],
                    [ 9.3288e-01,  8.4132e-04, -3.6019e-01, -6.0589e-01],
                    [-3.6009e-01, -2.1585e-02, -9.3267e-01,  1.2672e+00],
                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
                ])
            elif camera_name == "observation.images.camera_left":
                INTRINSIC = np.array([
                    [597.5103,   0.,      320.     ],
                    [  0.,      597.51025, 240.     ],
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [-0.7693, -0.5675,  0.2934,  0.4529],
                    [-0.3603,  0.0062, -0.9328,  0.5140],
                    [ 0.5275, -0.8234, -0.2093,  0.8179],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])
            elif camera_name == "observation.images.camera_right":
                INTRINSIC = np.array([
                    [565.74756,   0.,      320.     ],
                    [  0.,      565.74756, 240.     ],  
                    [  0.,        0.,        1.     ]
                ])
                EXTRINSIC = np.array([
                    [ 0.5391, -0.8344, -0.1147, -0.4520],
                    [-0.5300, -0.2302, -0.8162,  0.5700],
                    [ 0.6546,  0.5008, -0.5663,  0.7847],
                    [ 0.0000,  0.0000,  0.0000,  1.0000]
                ])

    assert INTRINSIC is not None and EXTRINSIC is not None, (
        f"camera_name {camera_name!r} not found or unsupported for dataset {dataset_name!r} "
    )
    return INTRINSIC, EXTRINSIC


if __name__ == "__main__":
    print("place_in_bread_in_basket" in ROBOMIND_SCENE4_SET)
    print(get_intrinsic_extrinsic("robomind/franka_3rgb", "place_in_bread_in_basket", "observation.images.camera_top", 0))