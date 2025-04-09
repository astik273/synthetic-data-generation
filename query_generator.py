from call_models import call_models

class QueryGenerator:
    def generate_queries(self, entity):
        system_prompt = "You are a helpful assistant generating queries for code entities."
        user_prompt = f"Generate a natural language query for the code entity: {entity}"
        response = call_models(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )
        return response.split('\n')