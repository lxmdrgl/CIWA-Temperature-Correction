"""Output order shared by training, evaluation, and visualization."""


def output_names(reg_output="canopy"):
    if reg_output == "canopy":
        return ("canopy",)
    if reg_output == "cwsi":
        return ("canopy", "upper_tail", "lower_tail")
    raise ValueError("reg_output must be canopy or cwsi")
