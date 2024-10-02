config = {
    "email": "",  # email used to log into 311 website
    "password": "",  # password used to log into 311 website
    "311_key": "",  # API key for checking status of open 311 service requests
    "model": "pretrained",  #'custom' == model trained on dot cam imgs, 'pretrained' == trained on COCO
    "runtime_hours": 8,
    "start_time": {"hour": 8, "minute": 30},
    "cameras": "",  # path to a csv file containing camera information
    "annotate_roi": True,  # flag indicating if the region of interest should be drawn on annotated images
    "post_summons_grace_hours": 1,  # amount of time camera will not be checked after summons issued
}
