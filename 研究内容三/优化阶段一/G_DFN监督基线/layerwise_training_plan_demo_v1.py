# -*- coding: utf-8 -*-

TRAINING_PLAN_METADATA = {
    "plan_name": "layerwise_training_plan_demo_v1",
    "updated_at": "2026-05-06",
    "description": "第一轮修改使用的8个规范层段训练计划，支持分层差异化训练参数。",
}

ALLOWED_LAYER_SURFACE_PAIR_KEYS = [
    "TOP_1100MS->T1",
    "T1->T2",
    "T2->T3",
    "T3->T4",
    "T4->T5",
    "T5->T6",
    "T6->T7",
    "T7->BOTTOM_3800MS",
]

LAYER_TRAINING_OVERRIDES = {
    "TOP_1100MS->T1": {
        "epochs": 5,
        "learning_rate": 8.0e-4,
        "center_positive_weight": 28.0,
        "count_positive_weight": 14.0,
        "calibration_window_weight": 0.40,
    },
    "T1->T2": {
        "epochs": 3,
    },
    "T2->T3": {
        "epochs": 4,
        "center_positive_weight": 26.0,
        "count_positive_weight": 13.0,
        "calibration_window_weight": 0.40,
    },
    "T3->T4": {
        "epochs": 5,
        "center_positive_weight": 30.0,
        "count_positive_weight": 16.0,
        "center_loss_weight": 3.0,
        "calibration_window_weight": 0.45,
    },
    "T4->T5": {
        "epochs": 4,
        "center_positive_weight": 28.0,
        "count_positive_weight": 14.0,
        "calibration_window_weight": 0.40,
    },
    "T5->T6": {
        "epochs": 6,
        "learning_rate": 8.0e-4,
        "center_positive_weight": 34.0,
        "count_positive_weight": 18.0,
        "center_loss_weight": 3.2,
        "calibration_window_weight": 0.50,
    },
    "T6->T7": {
        "epochs": 8,
        "learning_rate": 6.0e-4,
        "center_positive_weight": 40.0,
        "count_positive_weight": 22.0,
        "center_loss_weight": 3.5,
        "calibration_window_weight": 0.55,
    },
    "T7->BOTTOM_3800MS": {
        "epochs": 4,
        "learning_rate": 7.0e-4,
        "center_positive_weight": 20.0,
        "count_positive_weight": 10.0,
        "center_loss_weight": 2.2,
        "calibration_window_weight": 0.30,
    },
}
