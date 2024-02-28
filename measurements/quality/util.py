
def normalize_and_clamp(value, cap_value):
    if value > cap_value:
        value = cap_value
    normalized_value = value / cap_value
    return normalized_value