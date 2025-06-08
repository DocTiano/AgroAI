"""
Script to fetch Philippine location data from the PSGC API
and generate a JavaScript file with the data for use in the AgroAI application.

API Source: https://psgc.gitlab.io/api/
"""

import json
import requests
import os

# API endpoints
PROVINCES_URL = "https://psgc.gitlab.io/api/provinces.json"
DISTRICTS_URL = "https://psgc.gitlab.io/api/districts.json"  # For NCR districts
CITIES_URL = "https://psgc.gitlab.io/api/cities.json"
MUNICIPALITIES_URL = "https://psgc.gitlab.io/api/municipalities.json"
BARANGAYS_URL = "https://psgc.gitlab.io/api/barangays.json"

# Output file path
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "ph_locations.js")

def fetch_data(url):
    """Fetch data from the given URL"""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

def main():
    print("Fetching provinces data...")
    provinces_data = fetch_data(PROVINCES_URL)
    
    print("Fetching districts data (for NCR)...")
    districts_data = fetch_data(DISTRICTS_URL)
    
    print("Fetching cities data...")
    cities_data = fetch_data(CITIES_URL)
    
    print("Fetching municipalities data...")
    municipalities_data = fetch_data(MUNICIPALITIES_URL)
    
    print("Fetching barangays data...")
    barangays_data = fetch_data(BARANGAYS_URL)
    
    # Process provinces
    provinces = []
    for province in provinces_data:
        provinces.append({
            "name": province["name"],
            "code": province["code"]
        })
    
    # Add NCR districts as "provinces" for the dropdown
    for district in districts_data:
        if district["regionCode"] == "130000000":  # NCR
            provinces.append({
                "name": district["name"],
                "code": district["code"]
            })
    
    # Sort provinces by name
    provinces.sort(key=lambda x: x["name"])
    
    # Process cities and municipalities
    cities_municipalities = {}
    for city in cities_data:
        parent_code = city.get("provinceCode") or city.get("districtCode")
        if parent_code:
            if parent_code not in cities_municipalities:
                cities_municipalities[parent_code] = []
            cities_municipalities[parent_code].append({
                "name": city["name"],
                "code": city["code"]
            })
    
    for municipality in municipalities_data:
        parent_code = municipality.get("provinceCode")
        if parent_code:
            if parent_code not in cities_municipalities:
                cities_municipalities[parent_code] = []
            cities_municipalities[parent_code].append({
                "name": municipality["name"],
                "code": municipality["code"]
            })
    
    # Sort cities and municipalities by name
    for parent_code in cities_municipalities:
        cities_municipalities[parent_code].sort(key=lambda x: x["name"])
    
    # Process barangays
    barangays = {}
    for barangay in barangays_data:
        parent_code = barangay.get("cityCode") or barangay.get("municipalityCode")
        if parent_code:
            if parent_code not in barangays:
                barangays[parent_code] = []
            barangays[parent_code].append({
                "name": barangay["name"],
                "code": barangay["code"]
            })
    
    # Sort barangays by name
    for parent_code in barangays:
        barangays[parent_code].sort(key=lambda x: x["name"])
    
    # Create the JavaScript data structure
    js_data = {
        "provinces": provinces,
        "municipalities": cities_municipalities,
        "barangays": barangays
    }
    
    # Generate the JavaScript file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("/**\n")
        f.write(" * Philippine Locations Data\n")
        f.write(" * Generated from the PSGC API (https://psgc.gitlab.io/api/)\n")
        f.write(" * Contains provinces, cities/municipalities, and barangays in the Philippines\n")
        f.write(" */\n\n")
        f.write("const phLocations = ")
        f.write(json.dumps(js_data, indent=2, ensure_ascii=False))
        f.write(";\n")
    
    print(f"Successfully generated {OUTPUT_FILE}")
    print(f"- {len(provinces)} provinces/districts")
    print(f"- {sum(len(municipalities) for municipalities in cities_municipalities.values())} cities/municipalities")
    print(f"- {sum(len(barangay_list) for barangay_list in barangays.values())} barangays")

if __name__ == "__main__":
    main()
