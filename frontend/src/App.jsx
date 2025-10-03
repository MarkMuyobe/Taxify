import { useState } from "react";

function App() {
  const [location, setLocation] = useState("Downtown");
  const [timeOfDay, setTimeOfDay] = useState("Morning");
  const [prediction, setPrediction] = useState(null);
  const [modelType, setModelType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    // artificial wait between 1 and 5 seconds to simulate processing/network delay
    const delayMs = Math.floor(1000 + Math.random() * 4000);
    await new Promise((res) => setTimeout(res, delayMs));

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ location, time_of_day: timeOfDay }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Request failed");
      }

      const data = await res.json();
      setPrediction(data.predicted_wait_time);
      setModelType(data.model_type || null);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="card-header">
        <h1>Taxi Wait Time</h1>
        <p className="subtitle">Predict estimated wait time for a taxi</p>
      </header>

      <form onSubmit={handleSubmit} className="form-grid">
        <div className="field">
          <label className="field-label">Location</label>
          <select className="field-input" value={location} onChange={(e) => setLocation(e.target.value)}>
            <option>Downtown</option>
            <option>Airport</option>
            <option>Suburbs</option>
          </select>
        </div>

        <div className="field">
          <label className="field-label">Time of Day</label>
          <select className="field-input" value={timeOfDay} onChange={(e) => setTimeOfDay(e.target.value)}>
            <option>Morning</option>
            <option>Afternoon</option>
            <option>Evening</option>
            <option>Night</option>
          </select>
        </div>

        <div className="actions">
          <button className="btn primary" type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Wait Time"}
          </button>
        </div>
      </form>

      {error && <div className="error">Error: {error}</div>}

      {prediction !== null && (
        <div className="result-card">
          <div className="result-value">{Number(prediction).toFixed(2)}</div>
          <div className="result-label">Estimated minutes</div>
        </div>
      )}
      {modelType && (
        <div className="model-badge">Model: {modelType}</div>
      )}
    </div>
  );
}

export default App;
