
from src.ml_models.regression import predict_response_time

prediction = predict_response_time(
    query_length=12,
    context_length=1800,
    retrieved_chunks=5
)

print(f"Estimated response time: {prediction:.2f} seconds")
