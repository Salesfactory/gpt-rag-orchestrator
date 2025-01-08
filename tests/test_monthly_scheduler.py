import unittest
import azure.functions as func
from monthly_scheduler import main

class TestMonthlyScheduler(unittest.TestCase):
    def test_monthly_scheduler_trigger(self):
        # Setup
        timer = func.TimerRequest(
            schedule={},
            past_due=False
        )

        # Act
        response = main(timer)

        # Assert
        self.assertIsNone(response)  # Assuming the function returns None on success

if __name__ == '__main__':
    unittest.main() 