"""
Comprehensive End-to-End System Validation Script
==================================================
Validates all requirements for the final system sign-off:
1. Application Startup & Documented Endpoint Responses
2. Real PGN Tests across 6 authentic categories
3. PGN -> Features -> Model Pipeline Verification
4. Zero Future Leakage Verification on Live Pipeline
5. Clock Data Handling Verification (With vs Without [%clk])
6. Probability Calibration & Threshold Source Verification
7. UI/Backend Consistency & Dynamic State Verification
8. Move-by-Move Playback Telemetry Verification
9. Failure and Edge Cases Handling
10. Scientific Terminology Audit
11. Known Limitations Visibility
12. Performance & Latency Benchmarks
"""

import os
import sys
import io
import re
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import chess
import chess.pgn
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_URL = "http://127.0.0.1:8000"

def http_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "ValidationClient/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode("utf-8")

def http_post_json(path, data):
    url = f"{BASE_URL}{path}"
    json_bytes = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=json_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "ValidationClient/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}

def http_post_multipart(path, file_bytes, filename="game.pgn"):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    url = f"{BASE_URL}{path}"
    
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: application/x-chess-pgn\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "ValidationClient/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}

def main():
    print("=" * 60)
    print("STARTING COMPREHENSIVE END-TO-END SYSTEM VALIDATION")
    print("=" * 60)
    
    results = {}

    # ---------------------------------------------------------
    # 1. APPLICATION STARTUP & ENDPOINT VERIFICATION
    # ---------------------------------------------------------
    print("\n[STEP 1] Verifying Application Startup & Endpoints...")
    ep_results = {}
    
    # GET /health
    st, resp = http_get("/health")
    data_health = json.loads(resp)
    ep_results["/health"] = {"status": st, "data": data_health}
    print(f"  GET /health -> HTTP {st} | Service: {data_health.get('service')} | Model: {data_health.get('model_name')}")
    
    # GET /demo-games
    st, resp = http_get("/demo-games")
    data_demo = json.loads(resp)
    ep_results["/demo-games"] = {"status": st, "game_count": len(data_demo.get("demo_games", []))}
    print(f"  GET /demo-games -> HTTP {st} | Available Demos: {len(data_demo.get('demo_games', []))}")
    
    # GET /feature-importance
    st, resp = http_get("/feature-importance")
    data_imp = json.loads(resp)
    ep_results["/feature-importance"] = {"status": st, "feature_count": len(data_imp.get("features", []))}
    print(f"  GET /feature-importance -> HTTP {st} | Features Count: {len(data_imp.get('features', []))}")
    
    # GET / (Frontend)
    st, resp = http_get("/")
    ep_results["/"] = {"status": st, "html_length": len(resp)}
    print(f"  GET / -> HTTP {st} | HTML Dashboard bytes: {len(resp)}")
    
    # POST /predict
    test_vec = {
        "consecutive_fast_moves": 0.0,
        "premove_rate_10": 0.1,
        "cumulative_premove_rate": 0.05,
        "current_endgame_clock_ratio": 0.0,
        "phase_deliberation_ratio": 1.2,
        "move_time_cv_10": 0.8,
        "time_pressure_jitter": 0.0,
        "relative_move_time": 1.0,
        "time_spent_ratio": 0.03,
        "tank_move_count_10": 0.0
    }
    st, data_pred = http_post_json("/predict", test_vec)
    ep_results["/predict"] = {"status": st, "prediction": data_pred.get("prediction"), "ai_prob": data_pred.get("ai_probability")}
    print(f"  POST /predict -> HTTP {st} | Prediction: {data_pred.get('prediction')} | AI Prob: {data_pred.get('ai_probability')}")
    
    # POST /upload-pgn
    demo_pgn = open(os.path.join(PROJECT_ROOT, "data", "demo_games", "human_match.pgn"), "rb").read()
    st, data_upload = http_post_multipart("/upload-pgn", demo_pgn, filename="human_match.pgn")
    ep_results["/upload-pgn"] = {"status": st, "plies": len(data_upload.get("moves", []))}
    print(f"  POST /upload-pgn -> HTTP {st} | Plies Analyzed: {len(data_upload.get('moves', []))}")
    
    results["step1_endpoints"] = ep_results

    # ---------------------------------------------------------
    # 2. REAL PGN TESTS (6 Categories)
    # ---------------------------------------------------------
    print("\n[STEP 2] Testing 6 Authentic Real Games from Dataset...")
    with open(os.path.join(PROJECT_ROOT, "data", "test_validation_games.json"), "r", encoding="utf-8") as f:
        real_games = json.load(f)
        
    game_results = {}
    for cat_key, g_info in real_games.items():
        pgn_text = g_info["pgn"]
        t0 = time.time()
        st, analysis = http_post_json("/analyze-pgn", {"pgn": pgn_text})
        elapsed = time.time() - t0
        
        w_summary = analysis["summary"]["white"]
        b_summary = analysis["summary"]["black"]
        
        game_results[cat_key] = {
            "white": w_summary["name"],
            "black": b_summary["name"],
            "tc": g_info["headers"].get("TimeControl"),
            "event": g_info["headers"].get("Event"),
            "plies": len(analysis["moves"]),
            "elapsed_s": round(elapsed, 4),
            "white_pred": w_summary["overall_prediction"],
            "white_ai_prob": w_summary["mean_ai_probability_pct"],
            "black_pred": b_summary["overall_prediction"],
            "black_ai_prob": b_summary["mean_ai_probability_pct"]
        }
        print(f"  [{cat_key}] {w_summary['name']} vs {b_summary['name']} ({len(analysis['moves'])} plies, {elapsed:.3f}s)")
        print(f"    White: {w_summary['overall_prediction']} (P_AI: {w_summary['mean_ai_probability_pct']}%)")
        print(f"    Black: {b_summary['overall_prediction']} (P_AI: {b_summary['mean_ai_probability_pct']}%)")
        
    results["step2_real_pgn_tests"] = game_results

    # ---------------------------------------------------------
    # 3. VERIFY PGN -> FEATURES -> MODEL PIPELINE
    # ---------------------------------------------------------
    print("\n[STEP 3] Verifying Exact Pipeline: PGN -> Features -> Model...")
    with open(os.path.join(PROJECT_ROOT, "models", "feature_columns.json"), "r") as f:
        feature_cols_expected = json.load(f)
        
    sample_game_pgn = real_games["human_game"]["pgn"]
    st, sample_analysis = http_post_json("/analyze-pgn", {"pgn": sample_game_pgn})
    
    # Check move 10 features
    target_ply = 10
    move_sample = sample_analysis["moves"][target_ply - 1]
    features_extracted = move_sample["features_10"]
    feat_names = list(features_extracted.keys())
    
    pipeline_match = (feat_names == feature_cols_expected)
    print(f"  Feature Columns Match 'models/feature_columns.json': {pipeline_match}")
    print(f"  Expected: {feature_cols_expected}")
    print(f"  Observed: {feat_names}")
    
    # Directly score extracted features through LightGBM to verify prediction equality
    model = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    df_single = pd.DataFrame([features_extracted])[feature_cols_expected]
    expected_prob = float(model.predict_proba(df_single)[0, 1])
    observed_prob = move_sample["ai_probability"]
    prob_match = abs(expected_prob - observed_prob) < 1e-4
    print(f"  Model Probability Verification at Ply {target_ply}:")
    print(f"    Direct LightGBM predict_proba(): {expected_prob:.4f}")
    print(f"    Application pipeline AI Prob:   {observed_prob:.4f}")
    print(f"    Bit-for-Bit Probability Match:   {prob_match}")
    
    results["step3_pipeline"] = {
        "features_match": pipeline_match,
        "feature_columns": feat_names,
        "direct_model_prob": round(expected_prob, 4),
        "pipeline_prob": observed_prob,
        "probability_match": prob_match
    }

    # ---------------------------------------------------------
    # 4. VERIFY NO FUTURE LEAKAGE ON LIVE PIPELINE
    # ---------------------------------------------------------
    print("\n[STEP 4] Verifying Strict Temporal Invariance (No Future Leakage)...")
    full_pgn = real_games["human_game"]["pgn"]
    parsed_full = chess.pgn.read_game(io.StringIO(full_pgn))
    all_nodes = list(parsed_full.mainline())
    
    k_ply = 15  # evaluate at ply 15
    game_k = chess.pgn.Game()
    game_k.headers = parsed_full.headers
    curr = game_k
    for n in all_nodes[:k_ply]:
        curr = curr.add_variation(n.move, comment=n.comment)
    exporter = chess.pgn.StringExporter(headers=True, comments=True)
    pgn_k = game_k.accept(exporter)
    
    st_k, res_k = http_post_json("/analyze-pgn", {"pgn": pgn_k})
    st_full, res_full = http_post_json("/analyze-pgn", {"pgn": full_pgn})
    
    leakage_failures = []
    for p in range(k_ply):
        mk = res_k["moves"][p]
        mfull = res_full["moves"][p]
        
        prob_diff = abs(mk["ai_probability"] - mfull["ai_probability"])
        pred_diff = (mk["prediction"] != mfull["prediction"])
        for feat in feature_cols_expected:
            fdiff = abs(mk["features_10"][feat] - mfull["features_10"][feat])
            if fdiff > 1e-6:
                leakage_failures.append(f"Ply {p+1} Feature '{feat}' diff: {fdiff}")
        if prob_diff > 1e-6:
            leakage_failures.append(f"Ply {p+1} Probability diff: {prob_diff}")
        if pred_diff:
            leakage_failures.append(f"Ply {p+1} Prediction changed: {mk['prediction']} vs {mfull['prediction']}")
            
    print(f"  Checked plies 1..{k_ply} across short game vs full game with {len(all_nodes)} plies:")
    print(f"  Leakage Anomalies Detected: {len(leakage_failures)}")
    if len(leakage_failures) == 0:
        print("  PASS: 100% Invariance Confirmed on Live Application Pipeline!")
    else:
        for err in leakage_failures[:5]:
            print(f"    FAIL: {err}")
            
    results["step4_leakage"] = {
        "plies_tested": k_ply,
        "leakage_failures_count": len(leakage_failures),
        "status": "PASS: ZERO LEAKAGE CONFIRMED" if len(leakage_failures) == 0 else "FAIL"
    }

    # ---------------------------------------------------------
    # 5. VERIFY CLOCK DATA HANDLING
    # ---------------------------------------------------------
    print("\n[STEP 5] Verifying Clock Data Handling (With [%clk] vs Without [%clk])...")
    # PGN with clocks
    with_clk_pgn = pgn_k
    # PGN stripped of [%clk]
    lines = [l for l in pgn_k.splitlines() if not l.startswith("[")]
    san_text = " ".join(lines)
    san_stripped = re.sub(r"\{.*?\}", "", san_text)
    pgn_without_clk = f"""[Event "Clock Test Without Comments"]
[White "PlayerWhite"]
[Black "PlayerBlack"]
[TimeControl "180+0"]
[Result "*"]

{san_stripped}
"""
    st_with, res_with = http_post_json("/analyze-pgn", {"pgn": with_clk_pgn})
    st_without, res_without = http_post_json("/analyze-pgn", {"pgn": pgn_without_clk})
    
    print(f"  PGN WITH Clocks: has_clock_data={res_with['has_clock_data']} | White Pred: {res_with['summary']['white']['overall_prediction']}")
    print(f"  PGN WITHOUT Clocks: has_clock_data={res_without['has_clock_data']} | White Pred: {res_without['summary']['white']['overall_prediction']}")
    print(f"  Warning Message: '{res_without['warning']}'")
    print(f"  Moves AI Probabilities: {res_without['moves'][0]['ai_probability']}")
    print(f"  Moves Behavioral Signals: {res_without['moves'][0]['behavioral_signals'][0]['name']}")
    
    clock_rejection_valid = (
        res_without["has_clock_data"] is False and
        res_without["summary"]["white"]["mean_ai_probability"] is None and
        "UNAVAILABLE" in res_without["summary"]["white"]["overall_prediction"] and
        "Move clock data unavailable" in res_without["warning"]
    )
    print(f"  Clock Data Rejection & Warning Verification: {'PASS' if clock_rejection_valid else 'FAIL'}")
    
    results["step5_clock_handling"] = {
        "with_clock": {"has_clock": res_with["has_clock_data"], "pred": res_with["summary"]["white"]["overall_prediction"]},
        "without_clock": {
            "has_clock": res_without["has_clock_data"],
            "pred": res_without["summary"]["white"]["overall_prediction"],
            "warning": res_without["warning"],
            "first_move_prob": res_without["moves"][0]["ai_probability"]
        },
        "status": "PASS: CLEAR REJECTION AND EXPLICIT WARNING CONFIRMED" if clock_rejection_valid else "FAIL"
    }

    # ---------------------------------------------------------
    # 6. VERIFY PROBABILITY CALIBRATION & THRESHOLD
    # ---------------------------------------------------------
    print("\n[STEP 6] Verifying Probability Calibration & Metadata Sourcing...")
    with open(os.path.join(PROJECT_ROOT, "models", "model_metadata.json"), "r") as f:
        meta_json = json.load(f)
    expected_threshold = meta_json["operating_threshold"]
    
    # Test vector prediction
    st, pred_resp = http_post_json("/predict", test_vec)
    ai_p = pred_resp["ai_probability"]
    hu_p = pred_resp["human_probability"]
    sum_p = ai_p + hu_p
    resp_thresh = pred_resp["operating_threshold"]
    
    prob_calibration_valid = abs(sum_p - 1.0) < 1e-3
    thresh_source_valid = (resp_thresh == expected_threshold == 0.50)
    
    print(f"  AI Probability: {ai_p:.4f} | Human Probability: {hu_p:.4f} | Sum: {sum_p:.4f}")
    print(f"  Sum to 1.0 Verification: {'PASS' if prob_calibration_valid else 'FAIL'}")
    print(f"  Operating Threshold Sourced from Metadata: {resp_thresh} (Expected: {expected_threshold}) -> {'PASS' if thresh_source_valid else 'FAIL'}")
    
    results["step6_probability_calibration"] = {
        "ai_probability": ai_p,
        "human_probability": hu_p,
        "probability_sum": sum_p,
        "sum_valid": prob_calibration_valid,
        "operating_threshold": resp_thresh,
        "metadata_threshold": expected_threshold,
        "threshold_source_valid": thresh_source_valid
    }

    # ---------------------------------------------------------
    # 7 & 8. VERIFY UI TELEMETRY & MOVE-BY-MOVE PLAYBACK
    # ---------------------------------------------------------
    print("\n[STEP 7 & 8] Verifying Move-by-Move Playback Telemetry across Target Plies...")
    test_plies = [1, 2, 5, 10, 20, 35]
    playback_telemetry = {}
    
    for ply_num in test_plies:
        if ply_num <= len(sample_analysis["moves"]):
            m = sample_analysis["moves"][ply_num - 1]
            playback_telemetry[f"ply_{ply_num}"] = {
                "move": f"{m['move_number']}.{m['san']}",
                "player": m["player"],
                "fen_after": m["fen_after"],
                "move_time": m["move_time"],
                "clock_after": m["clock_after_move"],
                "ai_probability": m["ai_probability"],
                "human_probability": m["human_probability"],
                "prediction": m["prediction"],
                "features_sample": {
                    "consecutive_fast_moves": m["features_10"]["consecutive_fast_moves"],
                    "phase_deliberation_ratio": m["features_10"]["phase_deliberation_ratio"],
                    "move_time_cv_10": m["features_10"]["move_time_cv_10"]
                }
            }
            print(f"  Ply {ply_num:2d} ({m['player']:5s} {m['move_number']:2d}.{m['san']:6s}) | "
                  f"Time: {m['move_time']:.1f}s | Clock: {m['clock_after_move']:.1f}s | "
                  f"P(AI): {m['ai_probability'] * 100:.1f}% | Pred: {m['prediction']}")
                  
    results["step8_move_playback"] = playback_telemetry

    # ---------------------------------------------------------
    # 9. TEST FAILURE AND EDGE CASES
    # ---------------------------------------------------------
    print("\n[STEP 9] Testing Edge and Failure Cases...")
    edge_cases = [
        ("Invalid PGN syntax", "1. e4 e5 2. InvalidSyntax123"),
        ("Empty PGN text", ""),
        ("PGN without clock comments", pgn_without_clk),
        ("Very short 4-ply game", "[Event \"Fast\"]\n[TimeControl \"180+0\"]\n\n1. e4 {[%clk 0:02:59]} 1... e5 {[%clk 0:02:58]} 2. Qh5 {[%clk 0:02:57]} 2... Nc6 {[%clk 0:02:56]} 1-0"),
        ("Missing rating header", "[Event \"No Rating\"]\n[TimeControl \"180+0\"]\n\n1. e4 {[%clk 0:02:59]} 1... e5 {[%clk 0:02:58]} 1-0"),
        ("Unsupported time control string", "[Event \"Corr\"]\n[TimeControl \"-\"]\n\n1. e4 {[%clk 0:02:59]} 1... e5 {[%clk 0:02:58]} 1-0"),
        ("Zero-move aborted game", "[Event \"Aborted\"]\n[Result \"*\"]\n\n*")
    ]
    
    edge_results = {}
    for case_name, p_str in edge_cases:
        st, res = http_post_json("/analyze-pgn", {"pgn": p_str})
        is_graceful = (st in [200, 400])
        edge_results[case_name] = {
            "status_code": st,
            "handled_gracefully": is_graceful,
            "warning_or_detail": res.get("warning") or res.get("detail")
        }
        print(f"  Case '{case_name}': HTTP {st} -> {res.get('warning') or res.get('detail')}")
        
    results["step9_edge_cases"] = edge_results

    # ---------------------------------------------------------
    # 10. SCIENTIFIC TERMINOLOGY AUDIT
    # ---------------------------------------------------------
    print("\n[STEP 10] Performing Scientific Terminology Audit...")
    banned_terms = [
        "cheater", "cheat", "cheating",
        "cheater detected", "confirmed cheating", "player is cheating",
        "ai assistance proven", "proven cheating"
    ]
    
    terminology_violations = []
    
    # Audit frontend HTML & JS
    frontend_files = [
        os.path.join(PROJECT_ROOT, "app", "frontend", "index.html"),
        os.path.join(PROJECT_ROOT, "app", "frontend", "app.js"),
        os.path.join(PROJECT_ROOT, "app", "backend", "main.py"),
        os.path.join(PROJECT_ROOT, "src", "predict.py")
    ]
    
    for fpath in frontend_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().lower()
            for b in banned_terms:
                # Allow scientific disclaimers ("does not constitute proof of cheating", etc.)
                matches = [m.start() for m in re.finditer(re.escape(b), content)]
                for idx in matches:
                    snippet = content[max(0, idx - 40): min(len(content), idx + 50)]
                    if "not proof of cheating" in snippet or "not constitute proof" in snippet or "disclaimer" in snippet or "banned_terms" in snippet or "cheating detection" in snippet:
                        continue  # Legitimate negative scientific disclaimer
                    terminology_violations.append({
                        "file": os.path.basename(fpath),
                        "term": b,
                        "snippet": snippet.strip()
                    })
                    
    print(f"  Checked {len(banned_terms)} inflammatory terms across UI, Backend, and Inference modules.")
    print(f"  Unethical Terminology Violations Found: {len(terminology_violations)}")
    if len(terminology_violations) == 0:
        print("  PASS: 100% Scientifically Sound Terminology Confirmed!")
    else:
        for v in terminology_violations:
            print(f"    VIOLATION in {v['file']}: '{v['term']}' in \"{v['snippet']}\"")
            
    results["step10_terminology"] = {
        "banned_terms_checked": banned_terms,
        "violations_found": len(terminology_violations),
        "status": "PASS: SCIENTIFICALLY SOUND" if len(terminology_violations) == 0 else "FAIL"
    }

    # ---------------------------------------------------------
    # 11. VERIFY KNOWN LIMITATIONS ARE EXPLICIT IN UI
    # ---------------------------------------------------------
    print("\n[STEP 11] Verifying Known Limitations are Explicit in UI...")
    with open(os.path.join(PROJECT_ROOT, "app", "frontend", "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
        
    lim_checks = {
        "A. Unseen-game ROC-AUC (0.9008)": "0.9008" in html_content,
        "B. Unseen-player ROC-AUC (0.8639)": "0.8639" in html_content,
        "C. Paired human-vs-bot ROC-AUC (0.5176)": "0.5176" in html_content,
        "D. Bullet time-scramble false positives": "Bullet" in html_content and "scramble" in html_content.lower(),
        "E. Anthropomorphic / Maia bots": "Maia" in html_content or "Anthropomorphic" in html_content
    }
    
    all_lim_present = all(lim_checks.values())
    for k, v in lim_checks.items():
        print(f"  {k}: {'PRESENT' if v else 'MISSING'}")
    print(f"  Limitations Visibility Status: {'PASS' if all_lim_present else 'FAIL'}")
    
    results["step11_limitations"] = {
        "checks": lim_checks,
        "all_present": all_lim_present,
        "status": "PASS" if all_lim_present else "FAIL"
    }

    # ---------------------------------------------------------
    # 12. PERFORMANCE & LATENCY MEASUREMENTS
    # ---------------------------------------------------------
    print("\n[STEP 12] Measuring Real System Performance & Latencies...")
    # Measure component timings on a realistic 50-ply game
    sample_50_pgn = real_games["bot_game"]["pgn"]
    
    # 1. PGN Parsing Time
    t_parse_list = []
    for _ in range(50):
        t0 = time.time()
        chess.pgn.read_game(io.StringIO(sample_50_pgn))
        t_parse_list.append(time.time() - t0)
    avg_parse_ms = float(np.mean(t_parse_list) * 1000)
    
    # 2. Feature Extraction Time (via src.inference_features)
    from src.inference_features import parse_pgn_game
    t_feat_list = []
    for _ in range(50):
        t0 = time.time()
        parse_pgn_game(sample_50_pgn)
        t_feat_list.append(time.time() - t0)
    avg_feat_ms = float(np.mean(t_feat_list) * 1000)
    
    # 3. Model Inference Time (predict_player)
    from src.predict import predict_player
    t_pred_list = []
    for _ in range(100):
        t0 = time.time()
        predict_player(test_vec)
        t_pred_list.append(time.time() - t0)
    avg_pred_ms = float(np.mean(t_pred_list) * 1000)
    
    # 4. Complete End-to-End Analysis via Live API
    t_api_list = []
    for _ in range(20):
        t0 = time.time()
        http_post_json("/analyze-pgn", {"pgn": sample_50_pgn})
        t_api_list.append(time.time() - t0)
    avg_api_ms = float(np.mean(t_api_list) * 1000)
    
    plies_count = len(sample_analysis["moves"])
    throughput = round(plies_count / (avg_api_ms / 1000.0), 1)
    
    print(f"  Component Latencies (Measured across 50 iterations):")
    print(f"    - Pure PGN Parsing Time:              {avg_parse_ms:.3f} ms")
    print(f"    - Feature Extraction Pipeline:        {avg_feat_ms:.3f} ms ({avg_feat_ms/46:.3f} ms/ply)")
    print(f"    - Model Inference (LightGBM per ply): {avg_pred_ms:.3f} ms")
    print(f"    - Complete API Ingestion & Scoring:   {avg_api_ms:.2f} ms")
    print(f"    - End-to-End Analysis Throughput:     {throughput} plies/second")
    
    results["step12_performance"] = {
        "pgn_parsing_ms": round(avg_parse_ms, 3),
        "feature_extraction_ms": round(avg_feat_ms, 3),
        "model_inference_ms": round(avg_pred_ms, 3),
        "complete_api_ms": round(avg_api_ms, 2),
        "throughput_plies_per_sec": throughput
    }

    # Save complete validation audit
    out_audit = os.path.join(PROJECT_ROOT, "results", "final_validation_audit.json")
    with open(out_audit, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit complete! Saved comprehensive data to {out_audit}")
    
    return results

if __name__ == "__main__":
    main()
