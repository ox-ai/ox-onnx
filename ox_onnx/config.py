
# model root folder
OX_MODELS = ".ox_models"
ONNX_MODELS = "onnx_models"



# models id that are supported by ox_onnx
MODEL_IDS = ["sentence-transformers/all-MiniLM-L6-v2"]


# its a implementation map that maps MODEL_IDS to MODEL_ARCHITECTURE file
MODEL_INTERF_MAP = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "model_import": "ox_onnx.interface.sentence_transformers_all_MiniLM_L6_v2",
        "model_architecture": "sentence-transformers/all-MiniLM-L6-v2",
        "model_class": "InterfaceModel",
    }
}