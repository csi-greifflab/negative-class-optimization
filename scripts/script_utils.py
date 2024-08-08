"""
Utils for scripts.
"""

def get_input_dim_from_agpos(ag_pos: str) -> int:
    """
    Tmp solution to get the dimension based on the antigen name.
     - if HR2B -> 10*20 = 200
     - if HR2P -> 21*20 = 420
     - otherwise -> 11*20 = 220

    This is an adaptation, so that we can reuse the code for the
    experimental datasets from Brij and Porebski.
    """
    ag = ag_pos.split("_")[0]
    if ag == "HR2B":
        return 200
    elif ag == "HR2P":
        return 420
    elif ag == "HELP":
        return 380  # 19*20
    else:
        return 220
