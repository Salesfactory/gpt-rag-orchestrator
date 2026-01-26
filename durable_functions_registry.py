"""
Durable Functions registry - single source of truth for the DFApp instance.
All durable functions (orchestrators, activities, entities) should import 'app' from here.
"""

import azure.durable_functions as df
import azure.functions as func

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
