



def check_model_existence(model_path: str,model_files:list[str]) -> bool:
    """
    Checks if the model and its associated files exist in the specified directory.

    Args:
        model_path (str, optional): The path to the directory containing the model files.
        Defaults to MODEL_PATH.

    Returns:
        bool: True if all required files exist in the specified directory; False otherwise.
    """
    for model_file in model_files:
        if not (model_path / model_file).exists():
            return False
    #return all((model_path / fname).exists() for fname in model_files)
    return True
