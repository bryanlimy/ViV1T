TIERS = (
    "train",
    "validation",  # i.e.`oracle`
    "live_test_main",
    "live_test_bonus",
    "final_test_main",
    "final_test_bonus",
)

# mouse ID and their corresponding dataset ID
MOUSE_IDS = {
    # old mice
    "A": "dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "B": "dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "C": "dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    "D": "dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce",
    "E": "dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    # new mice
    "F": "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "G": "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "H": "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "I": "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "J": "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
}
OLD_MICE = ("A", "B", "C", "D", "E")
NEW_MICE = ("F", "G", "H", "I", "J")

MAX_FRAME = 300  # maximum number of frames in sensorium2023 dataset
