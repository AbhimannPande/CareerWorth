<<<<<<< HEAD
# test_enhanced_predictor.py
import sys
from pathlib import Path
from enhanced_predictor import EnhancedSalaryPredictor

def run_test_case(predictor, case_number, input_data, expected):
    """Execute and validate a single test case"""
    print(f"\n{'='*60}")
    print(f"TEST CASE {case_number}: {input_data['occupation']} (Age {input_data['age']}, {input_data['hours-per-week']} hrs/week)")
    print("-"*60)
    
    try:
        # Run prediction
        result = predictor.predict(input_data)
        
        # Display results
        print(f"RAW PREDICTION:\t{expected['currency']}{result['annual']:,.2f}/year")
        print(f"MONTHLY:\t{expected['currency']}{result['monthly']:,.2f}")
        print(f"HOURS USED:\t{result['hours_used']} (from {input_data['hours-per-week']})")
        print(f"AGE GROUP:\t{result['age_group']}")
        
        # Validate results
        passed = True
        if 'min_annual' in expected:
            if not (expected['min_annual'] <= result['annual'] <= expected['max_annual']):
                print(f"❌ Salary out of range: Expected {expected['min_annual']:,.0f}-{expected['max_annual']:,.0f}")
                passed = False
        
        if 'hours_used' in expected:
            if result['hours_used'] != expected['hours_used']:
                print(f"❌ Hour mismatch: Expected {expected['hours_used']}")
                passed = False
        
        if passed:
            print("✅ TEST PASSED")
        return passed
        
    except Exception as e:
        if expected.get('should_fail'):
            print(f"✅ TEST PASSED (Correctly rejected: {str(e)})")
            return True
        print(f"❌ TEST FAILED: {str(e)}")
        return False

def main():
    # Initialize predictor
    predictor = EnhancedSalaryPredictor()
    
    # Common base input for India
    base_input = {
        "native-country": "India",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "relationship": "Husband",
        "gender": "Male",
        "capital-gain": 0,
        "capital-loss": 0
    }
    
    # Realistic test cases for Indian salaries (in INR)
    test_cases = [
        # Young professional (Tech support)
        {
            "input": {**base_input, "age": 25, "occupation": "Tech-support", "hours-per-week": 40},
            "expected": {
                "min_annual": 300000,
                "max_annual": 500000,
                "hours_used": 40,
                "currency": "₹"
            }
        },
        # Peak career (Executive)
        {
            "input": {**base_input, "age": 45, "occupation": "Exec-managerial", "hours-per-week": 50},
            "expected": {
                "min_annual": 1200000,
                "max_annual": 2500000,
                "hours_used": 40,  # Should cap at 40
                "currency": "₹"
            }
        },
        # Early retiree (Clerical)
        {
            "input": {**base_input, "age": 67, "occupation": "Adm-clerical", "hours-per-week": 30},
            "expected": {
                "min_annual": 180000,
                "max_annual": 350000,
                "hours_used": 30,
                "currency": "₹"
            }
        },
        # Elderly worker (Handlers)
        {
            "input": {**base_input, "age": 85, "occupation": "Handlers-cleaners", "hours-per-week": 40},
            "expected": {
                "min_annual": 120000,
                "max_annual": 200000,
                "hours_used": 20,  # Should cap at 20 for elderly
                "currency": "₹"
            }
        },
        # Centenarian (Craft-repair)
        {
            "input": {**base_input, "age": 100, "occupation": "Craft-repair", "hours-per-week": 100},
            "expected": {
                "min_annual": 80000,
                "max_annual": 150000,
                "hours_used": 10,  # Should cap at 10
                "currency": "₹"
            }
        },
        # Invalid age (Should reject)
        {
            "input": {**base_input, "age": 17, "occupation": "Tech-support", "hours-per-week": 20},
            "expected": {
                "should_fail": True,
                "currency": "₹"
            }
        }
    ]

    # Run all tests
    print("🚀 STARTING COMPREHENSIVE SALARY PREDICTION TESTS (INDIA)")
    passed_count = 0
    
    for i, case in enumerate(test_cases, 1):
        if run_test_case(predictor, i, case["input"], case["expected"]):
            passed_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed_count}/{len(test_cases)} passed")
    if passed_count == len(test_cases):
        print("✅ ALL TESTS PASSED - MODEL IS WORKING CORRECTLY")
    else:
        print(f"⚠️ {len(test_cases)-passed_count} tests failed - check adjustments")

if __name__ == "__main__":
=======
# test_enhanced_predictor.py
import sys
from pathlib import Path
from enhanced_predictor import EnhancedSalaryPredictor

def run_test_case(predictor, case_number, input_data, expected):
    """Execute and validate a single test case"""
    print(f"\n{'='*60}")
    print(f"TEST CASE {case_number}: {input_data['occupation']} (Age {input_data['age']}, {input_data['hours-per-week']} hrs/week)")
    print("-"*60)
    
    try:
        # Run prediction
        result = predictor.predict(input_data)
        
        # Display results
        print(f"RAW PREDICTION:\t{expected['currency']}{result['annual']:,.2f}/year")
        print(f"MONTHLY:\t{expected['currency']}{result['monthly']:,.2f}")
        print(f"HOURS USED:\t{result['hours_used']} (from {input_data['hours-per-week']})")
        print(f"AGE GROUP:\t{result['age_group']}")
        
        # Validate results
        passed = True
        if 'min_annual' in expected:
            if not (expected['min_annual'] <= result['annual'] <= expected['max_annual']):
                print(f"❌ Salary out of range: Expected {expected['min_annual']:,.0f}-{expected['max_annual']:,.0f}")
                passed = False
        
        if 'hours_used' in expected:
            if result['hours_used'] != expected['hours_used']:
                print(f"❌ Hour mismatch: Expected {expected['hours_used']}")
                passed = False
        
        if passed:
            print("✅ TEST PASSED")
        return passed
        
    except Exception as e:
        if expected.get('should_fail'):
            print(f"✅ TEST PASSED (Correctly rejected: {str(e)})")
            return True
        print(f"❌ TEST FAILED: {str(e)}")
        return False

def main():
    # Initialize predictor
    predictor = EnhancedSalaryPredictor()
    
    # Common base input for India
    base_input = {
        "native-country": "India",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "relationship": "Husband",
        "gender": "Male",
        "capital-gain": 0,
        "capital-loss": 0
    }
    
    # Realistic test cases for Indian salaries (in INR)
    test_cases = [
        # Young professional (Tech support)
        {
            "input": {**base_input, "age": 25, "occupation": "Tech-support", "hours-per-week": 40},
            "expected": {
                "min_annual": 300000,
                "max_annual": 500000,
                "hours_used": 40,
                "currency": "₹"
            }
        },
        # Peak career (Executive)
        {
            "input": {**base_input, "age": 45, "occupation": "Exec-managerial", "hours-per-week": 50},
            "expected": {
                "min_annual": 1200000,
                "max_annual": 2500000,
                "hours_used": 40,  # Should cap at 40
                "currency": "₹"
            }
        },
        # Early retiree (Clerical)
        {
            "input": {**base_input, "age": 67, "occupation": "Adm-clerical", "hours-per-week": 30},
            "expected": {
                "min_annual": 180000,
                "max_annual": 350000,
                "hours_used": 30,
                "currency": "₹"
            }
        },
        # Elderly worker (Handlers)
        {
            "input": {**base_input, "age": 85, "occupation": "Handlers-cleaners", "hours-per-week": 40},
            "expected": {
                "min_annual": 120000,
                "max_annual": 200000,
                "hours_used": 20,  # Should cap at 20 for elderly
                "currency": "₹"
            }
        },
        # Centenarian (Craft-repair)
        {
            "input": {**base_input, "age": 100, "occupation": "Craft-repair", "hours-per-week": 100},
            "expected": {
                "min_annual": 80000,
                "max_annual": 150000,
                "hours_used": 10,  # Should cap at 10
                "currency": "₹"
            }
        },
        # Invalid age (Should reject)
        {
            "input": {**base_input, "age": 17, "occupation": "Tech-support", "hours-per-week": 20},
            "expected": {
                "should_fail": True,
                "currency": "₹"
            }
        }
    ]

    # Run all tests
    print("🚀 STARTING COMPREHENSIVE SALARY PREDICTION TESTS (INDIA)")
    passed_count = 0
    
    for i, case in enumerate(test_cases, 1):
        if run_test_case(predictor, i, case["input"], case["expected"]):
            passed_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed_count}/{len(test_cases)} passed")
    if passed_count == len(test_cases):
        print("✅ ALL TESTS PASSED - MODEL IS WORKING CORRECTLY")
    else:
        print(f"⚠️ {len(test_cases)-passed_count} tests failed - check adjustments")

if __name__ == "__main__":
>>>>>>> d3f8614 (Initial commit with CareerWorth app)
    main()