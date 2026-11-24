"""
EcoVision AI - Waste Disposal Guide & Category Knowledge Base
Author: College AI Mini Project Team
Categories:
  0: Organic
  1: Plastic
  2: Glass
  3: Paper
  4: Metal
"""

from typing import Dict, Any, Optional

WASTE_CATEGORIES = {
    0: {
        "id": 0,
        "name": "Organic",
        "icon": "🌱",
        "color_hex": "#2E7D32",       # Forest Green
        "color_bgr": (50, 125, 46),   # For OpenCV drawing (B, G, R)
        "bin_type": "Green Bin (Compost / Wet Waste)",
        "disposal_suggestion": (
            "Deposit in the green wet-waste bin or backyard compost pile. "
            "Suitable for fruit peels, vegetable leftovers, coffee grounds, and garden clippings."
        ),
        "handling_tips": [
            "Keep separate from plastic packaging and twist-ties.",
            "Avoid adding plastic stickers found on fruits and vegetables.",
            "If composting at home, balance green waste (nitrogen) with dry leaves (carbon)."
        ],
        "environmental_impact": (
            "Composting organic waste diverts methane-producing materials from landfills "
            "and creates nutrient-rich soil enhancer."
        )
    },
    1: {
        "id": 1,
        "name": "Plastic",
        "icon": "🧴",
        "color_hex": "#0288D1",       # Light Blue
        "color_bgr": (209, 136, 2),   # For OpenCV drawing (B, G, R)
        "bin_type": "Blue Bin (Dry Recyclables)",
        "disposal_suggestion": (
            "Rinse thoroughly to remove food or liquid residue. Flatten bottles to save volume. "
            "Deposit in your designated dry plastics recycling collection."
        ),
        "handling_tips": [
            "Check the resin identification code (PET 1, HDPE 2 are widely recycled).",
            "Remove caps or leave them screwed on according to local facility rules.",
            "Keep thin film wraps separate if single-stream sorting excludes soft plastics."
        ],
        "environmental_impact": (
            "Recycling plastic reduces petroleum consumption and prevents long-lasting microplastic "
            "contamination in aquatic and terrestrial ecosystems."
        )
    },
    2: {
        "id": 2,
        "name": "Glass",
        "icon": "🍾",
        "color_hex": "#00897B",       # Teal
        "color_bgr": (123, 137, 0),   # For OpenCV drawing (B, G, R)
        "bin_type": "Yellow / Cyan Bin (Glass Collection)",
        "disposal_suggestion": (
            "Rinse bottles and jars clean. Remove metal or plastic lids and sort them separately. "
            "Deposit in designated glass drop-off banks or curbside bins."
        ),
        "handling_tips": [
            "Do not mix mirror glass, light bulbs, or cookware (Pyrex) with container glass.",
            "Wrap broken glass securely in newspaper or cardboard to protect sanitation workers.",
            "Glass can be recycled endlessly without quality degradation."
        ],
        "environmental_impact": (
            "Recycling 1 ton of glass saves over a ton of natural raw materials and cuts energy consumption by 30%."
        )
    },
    3: {
        "id": 3,
        "name": "Paper",
        "icon": "📄",
        "color_hex": "#F57C00",       # Amber / Orange
        "color_bgr": (0, 124, 245),   # For OpenCV drawing (B, G, R)
        "bin_type": "Blue / Paper Bin (Dry Paper & Cardboard)",
        "disposal_suggestion": (
            "Keep dry and clean. Flatten cardboard cartons and newspaper stacks. "
            "Deposit in clean paper recycling receptacles."
        ),
        "handling_tips": [
            "Greasy pizza boxes and soiled food packaging cannot be recycled with clean paper.",
            "Remove plastic tape, bubble wrap, and non-paper fasteners where feasible.",
            "Shredded paper should be bagged if required by your municipal hauler."
        ],
        "environmental_impact": (
            "Recycling paper saves trees, reduces industrial water usage by 58%, and conserves landfill space."
        )
    },
    4: {
        "id": 4,
        "name": "Metal",
        "icon": "🥫",
        "color_hex": "#78909C",       # Slate Gray
        "color_bgr": (156, 144, 120), # For OpenCV drawing (B, G, R)
        "bin_type": "Grey / Silver Bin (Metal Recyclables)",
        "disposal_suggestion": (
            "Empty and rinse aluminum beverage cans and steel/tin food cans. "
            "Place in designated dry metal recycling containers."
        ),
        "handling_tips": [
            "Crush aluminum beverage cans to optimize storage space.",
            "Empty aerosol cans completely before recycling (do not puncture).",
            "Scrap metal, wires, or heavy iron should be taken to dedicated scrap metal recyclers."
        ],
        "environmental_impact": (
            "Recycling aluminum saves 95% of the energy needed to produce new metal from bauxite ore."
        )
    }
}

# Municipality Segregation Disclaimer
DISCLAIMER_TEXT = (
    "⚠️ Disclaimer: Waste segregation categories and processing standards may vary across "
    "different municipalities, waste management authorities, and educational campuses. "
    "Always check your local city or institutional recycling protocols for region-specific guidelines."
)

NAME_TO_ID = {
    "organic": 0,
    "plastic": 1,
    "glass": 2,
    "paper": 3,
    "metal": 4
}


def get_category_info(identifier: Any) -> Optional[Dict[str, Any]]:
    """
    Retrieve full category information by either integer class ID (0-4) or string name.
    """
    if isinstance(identifier, int):
        return WASTE_CATEGORIES.get(identifier)
    elif isinstance(identifier, str):
        cleaned = identifier.strip().lower()
        cat_id = NAME_TO_ID.get(cleaned)
        if cat_id is not None:
            return WASTE_CATEGORIES.get(cat_id)
    return None


def get_all_categories() -> Dict[int, Dict[str, Any]]:
    """Return dictionary of all five waste categories."""
    return WASTE_CATEGORIES


def get_disclaimer() -> str:
    """Return standard waste segregation disclaimer."""
    return DISCLAIMER_TEXT
