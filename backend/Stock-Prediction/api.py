from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from chart_generator import generate_forecast_chart, generate_health_check_chart

app = FastAPI(title="Fish Stock Analysis API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fish Stock Analysis API! Now with interactive charts!"}

@app.get("/forecast_interactive",
         summary="Get Interactive Stock Forecast Chart",
         description="Returns a Plotly chart as JSON. A frontend can use this to render an interactive plot.")
def get_interactive_forecast_chart():
    """
    Generates and returns the interactive Plotly forecast chart as a JSON object.
    """
    chart_json = generate_forecast_chart()
    return Response(content=chart_json, media_type="application/json")

@app.get("/health-check",
         summary="Get Overfishing Status Chart",
         description="Returns a PNG image comparing total catch vs. stock per year.")
def get_health_check_chart():
    """
    Generates and returns the overfishing health check chart as a static image.
    """
    image_bytes = generate_health_check_chart()
    return StreamingResponse(image_bytes, media_type="image/png")