def parking_insights(occupied, empty):
    total = occupied + empty

    if total == 0:
        return {
            "total": 0,
            "occupied": 0,
            "available": 0,
            "occupancy": 0.0,
            "congestion": "No data",
            "recommendation": "Upload a valid parking image."
        }

    occupancy = (occupied / total) * 100
    available = empty

    if occupancy < 40:
        congestion = "Low"
        recommendation = "Parking availability is good — proceed to park."
    elif occupancy <= 75:
        congestion = "Moderate"
        recommendation = "Parking is moderately occupied — available slots remain."
    else:
        congestion = "High"
        recommendation = "Parking is highly occupied — consider another area."

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy": occupancy,
        "congestion": congestion,
        "recommendation": recommendation,
    }
