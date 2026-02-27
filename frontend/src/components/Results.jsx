import './Results.css'

function Results({ prediction }) {
    if (!prediction) return null

    const { activity, confidence, probabilities } = prediction

    // Sort probabilities for bar chart
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1])

    return (
        <div className="results glass-card">
            <h2>Classification Results</h2>

            <div className="prediction-card">
                <div className="activity-display">
                    <h3>{activity.replace(/_/g, ' ')}</h3>
                    <div className="confidence-badge">
                        {(confidence * 100).toFixed(1)}% Confidence
                    </div>
                </div>

                <div className="confidence-bar">
                    <div
                        className="confidence-fill"
                        style={{ width: `${confidence * 100}%` }}
                    ></div>
                </div>
            </div>

            <div className="probabilities-section">
                <h4>Probability Distribution</h4>
                <div className="probability-bars">
                    {sortedProbs.map(([act, prob]) => (
                        <div key={act} className="probability-item">
                            <div className="probability-label">
                                <span className="activity-name">{act.replace(/_/g, ' ')}</span>
                                <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="probability-bar-bg">
                                <div
                                    className={`probability-bar ${act === activity ? 'active' : ''}`}
                                    style={{ width: `${prob * 100}%` }}
                                ></div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default Results
