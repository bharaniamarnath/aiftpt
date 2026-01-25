def categorize_offensive_player(positions):
    if any(pos in positions for pos in ['ST', 'CF', 'LS', 'RS']):
        return 'Striker'
    elif any(pos in positions for pos in ['RW']):
        return 'Right Winger'
    elif any(pos in positions for pos in ['LW']):
        return 'Left Winger'
    return None
