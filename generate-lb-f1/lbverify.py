import os
import zipfile
import json
import shutil
import glob

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Get the directory where this script is currently located
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Automatically find the .zip file in the same directory
available_zips = glob.glob(os.path.join(WORKSPACE_DIR, "*.zip"))

if not available_zips:
    raise FileNotFoundError(f"⚠️ No master zip file found in {WORKSPACE_DIR}. Please place it next to this script!")

MASTER_ZIP_PATH = available_zips[0]
EXTRACT_DIR = os.path.join(WORKSPACE_DIR, "F1_Hackathon_Submissions")
HTML_OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "f1_leaderboard.html")

def process_submissions():
    # We will use a dictionary to track the highest score per participant.
    # We use a unique ID as the key, but we'll also check names and reg numbers.
    best_submissions = {}
    
    print(f"📦 Extracting Master Zip: {os.path.basename(MASTER_ZIP_PATH)}...")
    with zipfile.ZipFile(MASTER_ZIP_PATH, 'r') as master_zip:
        master_zip.extractall(EXTRACT_DIR)

    print("🔍 Scanning participant submissions...\n")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith('.zip'):
                submission_zip_path = os.path.join(root, file)
                temp_extract = os.path.join(root, file.replace('.zip', '_temp'))
                
                try:
                    # 1. Unzip the participant's submission
                    with zipfile.ZipFile(submission_zip_path, 'r') as sub_zip:
                        sub_zip.extractall(temp_extract)
                    
                    # 2. Locate metrics.json inside the extracted folder
                    metrics_path = os.path.join(temp_extract, 'metrics.json')
                    
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            data = json.load(f)
                            
                            raw_name = data.get("participant_name", "Unknown Driver").strip()
                            raw_reg = data.get("registration_number", "N/A").strip()
                            current_score = float(data.get("composite_score", 0.0))
                            
                            current_entry = {
                                "name": raw_name,
                                "registration_number": raw_reg,
                                "score": current_score,
                                "model": data.get("best_model", "Unknown Chassis")
                            }
                            
                            # 3. Deduplication Logic: Check if we've seen this name OR reg number
                            found_match = False
                            for existing_id, existing_entry in best_submissions.items():
                                match_reg = raw_reg.upper() == existing_entry["registration_number"].upper() and raw_reg != "N/A"
                                match_name = raw_name.upper() == existing_entry["name"].upper()
                                
                                if match_reg or match_name:
                                    found_match = True
                                    # If the new score is higher, overwrite the old one
                                    if current_score > existing_entry["score"]:
                                        print(f"📈 NEW PERSONAL BEST! {raw_name} improved from {existing_entry['score']} to {current_score}")
                                        best_submissions[existing_id] = current_entry
                                    else:
                                        print(f"➖ Ignored lower/equal score ({current_score}) for {raw_name}")
                                    break
                            
                            # If they are completely new, add them to the dictionary
                            if not found_match:
                                # Create an ID for dictionary storage
                                new_id = raw_reg if raw_reg != "N/A" else raw_name 
                                best_submissions[new_id] = current_entry
                                print(f"✅ Processed new driver: {raw_name} (Score: {current_score})")
                                
                    else:
                        print(f"⚠️ No metrics.json found in {file}")
                        
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")
                finally:
                    # 4. Clean up the temporary folder
                    if os.path.exists(temp_extract):
                        shutil.rmtree(temp_extract)

    # 5. Convert the dictionary into a list and sort by score (Highest = Rank 1)
    leaderboard = list(best_submissions.values())
    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    return leaderboard

def generate_f1_html(leaderboard):
    print("\n🎨 Generating F1-Themed HTML Leaderboard...")
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>F1 RL Hackathon Leaderboard</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,700;1,700&display=swap');
            
            body {
                background-color: #15151e; /* Dark tarmac */
                color: #ffffff;
                font-family: 'Titillium Web', sans-serif;
                margin: 0;
                padding: 40px;
            }
            .header-container {
                text-align: center;
                margin-bottom: 40px;
            }
            h1 {
                color: #e10600; /* Official F1 Red */
                text-transform: uppercase;
                font-style: italic;
                font-size: 3rem;
                letter-spacing: 2px;
                margin: 0;
            }
            h2 {
                color: #888;
                text-transform: uppercase;
                font-size: 1.2rem;
                letter-spacing: 5px;
                margin-top: 5px;
            }
            table {
                width: 90%;
                max-width: 1000px;
                margin: 0 auto;
                border-collapse: collapse;
                background-color: #1e1e2f;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                padding: 18px 25px;
                text-align: left;
                border-bottom: 1px solid #2a2a3f;
            }
            th {
                background-color: #e10600;
                color: white;
                text-transform: uppercase;
                font-weight: 700;
                font-size: 1.1rem;
                letter-spacing: 1px;
            }
            tr:hover {
                background-color: #2a2a3f;
                transition: background-color 0.2s ease;
            }
            .rank {
                font-size: 1.5rem;
                font-weight: bold;
                text-align: center;
            }
            .rank-1 { color: #ffd700; } /* Gold */
            .rank-2 { color: #c0c0c0; } /* Silver */
            .rank-3 { color: #cd7f32; } /* Bronze */
            
            .score {
                font-weight: bold;
                color: #00ff7f; /* Spring green for positive points */
                font-size: 1.2rem;
            }
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>🏁 Live Grand Prix Standings 🏁</h1>
            <h2>Autonomous Racing Championship</h2>
        </div>
        
        <table>
            <tr>
                <th>Pos</th>
                <th>Driver Name</th>
                <th>Reg Number</th>
                <th>Chassis (Model)</th>
                <th>Composite Score</th>
            </tr>
    """

    for i, entry in enumerate(leaderboard):
        rank = i + 1
        
        if rank == 1:
            pos_display = "🥇 1"
            row_class = "rank-1"
        elif rank == 2:
            pos_display = "🥈 2"
            row_class = "rank-2"
        elif rank == 3:
            pos_display = "🥉 3"
            row_class = "rank-3"
        else:
            pos_display = str(rank)
            row_class = ""

        html_content += f"""
            <tr>
                <td class="rank {row_class}">{pos_display}</td>
                <td style="font-weight: bold; font-size: 1.1rem;">{entry['name']}</td>
                <td style="color: #aaa;">{entry['registration_number']}</td>
                <td style="font-style: italic;">{entry['model']}</td>
                <td class="score">{entry['score']:,.1f}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(HTML_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"🏆 SUCCESS! Open this file in your browser to see the results: \n👉 {HTML_OUTPUT_PATH}")

if __name__ == "__main__":
    final_rankings = process_submissions()
    if final_rankings:
        generate_f1_html(final_rankings)
    else:
        print("⚠️ No valid submissions were found to rank.")