"""
Full integration test suite for the Student Performance Predictor.
Run with:  python tests/test_app.py
Requires Flask server to be running on port 5000.
"""
import json
import requests

BASE = 'http://127.0.0.1:5000'
PASS_COUNT = 0
FAIL_COUNT = 0


def test(name, method, url, body=None, expect_status=200, expect_key=None, expect_val=None):
    global PASS_COUNT, FAIL_COUNT
    try:
        if method == 'GET':
            r = requests.get(url, timeout=5)
        else:
            r = requests.post(url, json=body, timeout=5)

        ok = (r.status_code == expect_status)
        data = r.json()

        if ok and expect_key and expect_val is not None:
            ok = (data.get(expect_key) == expect_val)
        elif ok and expect_key and expect_val is None:
            ok = (expect_key in data)

        label = 'PASS' if ok else 'FAIL'
        if ok:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
            print(f'  [{label}] {name}')
            print(f'         HTTP {r.status_code} | got: {json.dumps(data)[:240]}')
            return

        print(f'  [{label}] {name}')

    except Exception as e:
        FAIL_COUNT += 1
        print(f'  [FAIL] {name}  →  {type(e).__name__}: {e}')


# ── Shared fixtures ────────────────────────────────────────────────────────────

PASS_STUDENT = dict(
    school='GP', sex='F', age=16, address='U', famsize='GT3', Pstatus='T',
    Medu=3, Fedu=3, Mjob='teacher', Fjob='services', reason='course', guardian='mother',
    traveltime=1, studytime=3, failures=0, schoolsup='no', famsup='yes',
    paid='no', activities='yes', nursery='yes', higher='yes', internet='yes',
    romantic='no', famrel=4, freetime=3, goout=2, Dalc=1, Walc=1, health=4,
    absences=2, G1=14, G2=13
)

FAIL_STUDENT = dict(
    school='MS', sex='M', age=20, address='R', famsize='LE3', Pstatus='A',
    Medu=0, Fedu=1, Mjob='at_home', Fjob='other', reason='home', guardian='other',
    traveltime=4, studytime=1, failures=3, schoolsup='no', famsup='no',
    paid='no', activities='no', nursery='no', higher='no', internet='no',
    romantic='yes', famrel=1, freetime=5, goout=5, Dalc=5, Walc=5, health=1,
    absences=40, G1=3, G2=2
)

# ── Tests ──────────────────────────────────────────────────────────────────────

print()
print('=' * 56)
print('  STUDENT PERFORMANCE PREDICTOR — TEST SUITE')
print('=' * 56)

# 1. Route availability
print('\n[ Routes ]')

# GET / returns HTML — test status + content type, not JSON
r_home = requests.get(f'{BASE}/', timeout=5)
if r_home.status_code == 200 and 'text/html' in r_home.headers.get('Content-Type', ''):
    PASS_COUNT += 1
    print('  [PASS] GET / → 200 HTML')
else:
    FAIL_COUNT += 1
    print(f'  [FAIL] GET / → expected 200 HTML, got {r_home.status_code} ({r_home.headers.get("Content-Type")})')

test('GET /model-stats → 200',              'GET', f'{BASE}/model-stats')
test('model-stats has metrics key',         'GET', f'{BASE}/model-stats', expect_key='metrics')
test('model-stats has feature_importances', 'GET', f'{BASE}/model-stats', expect_key='feature_importances')
test('model-stats has dataset_info',        'GET', f'{BASE}/model-stats', expect_key='dataset_info')

# 2. Core prediction — passing student
print('\n[ Prediction: Strong Pass ]')
test('response has prediction key',  'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='prediction')
test('predicts Pass',                'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='prediction',  expect_val='Pass')
test('pass=1',                       'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='pass',        expect_val=1)
test('probability is present',       'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='probability')
test('risk_band is present',         'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='risk_band')
test('insights list present',        'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='insights')
test('top_features list present',    'POST', f'{BASE}/predict', PASS_STUDENT, expect_key='top_features')

# 3. Core prediction — failing student
print('\n[ Prediction: Strong Fail ]')
test('predicts Fail',  'POST', f'{BASE}/predict', FAIL_STUDENT, expect_key='prediction', expect_val='Fail')
test('pass=0',         'POST', f'{BASE}/predict', FAIL_STUDENT, expect_key='pass',       expect_val=0)

# 4. Borderline student (grade just on pass threshold)
print('\n[ Borderline Student ]')
borderline = {**PASS_STUDENT, 'G1': 10, 'G2': 10, 'studytime': 2, 'failures': 1}
test('borderline → has prediction',  'POST', f'{BASE}/predict', borderline, expect_key='prediction')
test('borderline → has probability', 'POST', f'{BASE}/predict', borderline, expect_key='probability')

# 5. Edge cases — input boundaries
print('\n[ Edge Cases ]')
test('age=15 (min)',  'POST', f'{BASE}/predict', {**PASS_STUDENT, 'age': 15},    expect_key='prediction')
test('age=22 (max)',  'POST', f'{BASE}/predict', {**PASS_STUDENT, 'age': 22},    expect_key='prediction')
test('absences=0',    'POST', f'{BASE}/predict', {**PASS_STUDENT, 'absences': 0},expect_key='prediction')
test('absences=75',   'POST', f'{BASE}/predict', {**PASS_STUDENT, 'absences': 75, 'G1': 8, 'G2': 7}, expect_key='prediction')
test('G1=0, G2=0',    'POST', f'{BASE}/predict', {**FAIL_STUDENT, 'G1': 0, 'G2': 0},  expect_key='prediction')
test('G1=20, G2=20',  'POST', f'{BASE}/predict', {**PASS_STUDENT, 'G1': 20, 'G2': 20}, expect_key='prediction')
test('all sliders max','POST', f'{BASE}/predict', {**PASS_STUDENT, 'goout': 5, 'Dalc': 5, 'Walc': 5, 'freetime': 5, 'famrel': 5, 'health': 5}, expect_key='prediction')
test('all sliders min','POST', f'{BASE}/predict', {**PASS_STUDENT, 'goout': 1, 'Dalc': 1, 'Walc': 1, 'freetime': 1, 'famrel': 1, 'health': 1}, expect_key='prediction')
test('failures=0',    'POST', f'{BASE}/predict', {**PASS_STUDENT, 'failures': 0}, expect_key='prediction')
test('failures=3',    'POST', f'{BASE}/predict', {**PASS_STUDENT, 'failures': 3}, expect_key='prediction')

# 6. Error handling
print('\n[ Error Handling ]')
test('empty body → 400',           'POST', f'{BASE}/predict', {},                    expect_status=400, expect_key='error')
test('missing fields → 422',       'POST', f'{BASE}/predict', {'school': 'GP'},      expect_status=422, expect_key='error')
test('invalid school → 422',       'POST', f'{BASE}/predict', {**PASS_STUDENT, 'school': 'ZZ'},    expect_status=422, expect_key='error')
test('invalid sex → 422',          'POST', f'{BASE}/predict', {**PASS_STUDENT, 'sex': 'X'},        expect_status=422, expect_key='error')
test('invalid Mjob → 422',         'POST', f'{BASE}/predict', {**PASS_STUDENT, 'Mjob': 'astronaut'}, expect_status=422, expect_key='error')
test('invalid guardian → 422',     'POST', f'{BASE}/predict', {**PASS_STUDENT, 'guardian': 'uncle'}, expect_status=422, expect_key='error')

# 7. Risk band coverage
print('\n[ Risk Band Coverage ]')

def get_result(student):
    r = requests.post(f'{BASE}/predict', json=student, timeout=5).json()
    return r.get('risk_band', '?'), r.get('probability', 0)

low_risk  = {**PASS_STUDENT, 'G1': 18, 'G2': 18, 'failures': 0, 'absences': 0, 'studytime': 4}
very_high = FAIL_STUDENT

band_low,  prob_low  = get_result(low_risk)
band_base, prob_base = get_result(PASS_STUDENT)
band_vhi,  prob_vhi  = get_result(very_high)

print(f'  [INFO] Strong pass  → {band_low:<18} ({prob_low}%)')
print(f'  [INFO] Base pass    → {band_base:<18} ({prob_base}%)')
print(f'  [INFO] Strong fail  → {band_vhi:<18} ({prob_vhi}%)')

# Manual assertions on risk band logic
test_rb = requests.post(f'{BASE}/predict', json=low_risk, timeout=5).json()
risk_ok = test_rb.get('risk_band') == 'Low Risk'
if risk_ok:
    PASS_COUNT += 1
    print(f'  [PASS] High-confidence pass → "Low Risk"')
else:
    FAIL_COUNT += 1
    print(f'  [FAIL] Expected "Low Risk", got "{test_rb.get("risk_band")}"')

test_rb2 = requests.post(f'{BASE}/predict', json=FAIL_STUDENT, timeout=5).json()
risk_ok2 = test_rb2.get('risk_band') == 'Very High Risk'
if risk_ok2:
    PASS_COUNT += 1
    print(f'  [PASS] Low-confidence pass  → "Very High Risk"')
else:
    FAIL_COUNT += 1
    print(f'  [FAIL] Expected "Very High Risk", got "{test_rb2.get("risk_band")}"')

# ── Summary ────────────────────────────────────────────────────────────────────
total = PASS_COUNT + FAIL_COUNT
print()
print('=' * 56)
print(f'  {PASS_COUNT}/{total} tests passed  |  {FAIL_COUNT} failed')
print('=' * 56)
print()

if FAIL_COUNT > 0:
    exit(1)
