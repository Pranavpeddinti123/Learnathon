import { useState, useMemo } from 'react'
import './ManualInput.css'

function ManualInput({ onPredict, loading }) {
    const [rows, setRows] = useState(10)
    const [data, setData] = useState([])
    const [activePreset, setActivePreset] = useState('RANDOM')

    // Initialize with random data on mount
    useState(() => {
        setData(generatePatternData('RANDOM'))
    }, [])

    function generatePatternData(activityType) {
        const newData = []
        const timesteps = 128

        const configs = {
            WALKING: {
                bodyAcc: [-0.015, -0.01, 0.0],
                gyro: [-0.25, 0.14, -0.01],
                totalAcc: [0.97, -0.33, 0.0], // Neutral posture
                amp: 0.45, freq: 2.0
            },
            WALKING_UPSTAIRS: {
                bodyAcc: [-0.01, -0.01, -0.1],
                gyro: [-0.5, 0.1, -0.02],
                totalAcc: [0.96, -0.40, -0.15], // Clear forward lean
                amp: 0.35, freq: 2.5
            },
            WALKING_DOWNSTAIRS: {
                bodyAcc: [0.03, -0.03, 0.0],
                gyro: [-0.1, 0.0, -0.02],
                totalAcc: [1.02, -0.35, 0.10], // Impact/Vertical posture
                amp: 0.7, freq: 1.6
            },
            SITTING: {
                bodyAcc: [0.0, 0.0, 0.0],
                gyro: [0.0, 0.0, 0.0],
                totalAcc: [1.02, -0.02, 0.05],
                noise: 0.002
            },
            STANDING: {
                bodyAcc: [0.0, 0.0, 0.0],
                gyro: [0.0, 0.0, 0.0],
                totalAcc: [0.98, -0.55, 0.15],
                noise: 0.005
            },
            LAYING: {
                bodyAcc: [-0.02, 0.0, 0.0],
                gyro: [0.0, -0.1, 0.1],
                totalAcc: [-0.2, 0.8, 0.6],
                noise: 0.012
            }
        }

        for (let i = 0; i < timesteps; i++) {
            const row = []
            const t = (i / 50) * 2 * Math.PI // 50Hz sampling

            if (configs[activityType]) {
                const c = configs[activityType]

                if (activityType.startsWith('WALKING')) {
                    const phase = c.phase || 0
                    const wave = Math.sin(t * c.freq + phase)
                    const gWave = Math.cos(t * c.freq + phase)
                    const jitter = () => (Math.random() - 0.5) * 0.05

                    // Body Acc (Clear movement)
                    row.push((c.bodyAcc[0] + wave * c.amp + jitter()).toFixed(4))
                    row.push((c.bodyAcc[1] + gWave * c.amp + jitter()).toFixed(4))
                    row.push((c.bodyAcc[2] + wave * c.amp * 0.5 + jitter()).toFixed(4))

                    // Gyro (Clear rotation)
                    row.push((c.gyro[0] + wave * 1.5).toFixed(4))
                    row.push((c.gyro[1] + gWave * 1.0).toFixed(4))
                    row.push((c.gyro[2] + wave * 0.5).toFixed(4))

                    // Total Acc (Matched oscillation)
                    row.push((c.totalAcc[0] + wave * c.amp + jitter()).toFixed(4))
                    row.push((c.totalAcc[1] + gWave * c.amp + jitter()).toFixed(4))
                    row.push((c.totalAcc[2] + wave * c.amp * 0.5 + jitter()).toFixed(4))
                } else {
                    // Stationary modes: Realistic tremor noise
                    const n = () => (Math.random() - 0.5) * c.noise

                    row.push((c.bodyAcc[0] + n()).toFixed(4))
                    row.push((c.bodyAcc[1] + n()).toFixed(4))
                    row.push((c.bodyAcc[2] + n()).toFixed(4))

                    row.push((c.gyro[0] + n() * 5).toFixed(4))
                    row.push((c.gyro[1] + n() * 5).toFixed(4))
                    row.push((c.gyro[2] + n() * 5).toFixed(4))

                    row.push((c.totalAcc[0] + n()).toFixed(4))
                    row.push((c.totalAcc[1] + n()).toFixed(4))
                    row.push((c.totalAcc[2] + n()).toFixed(4))
                }
            } else {
                for (let j = 0; j < 9; j++) {
                    row.push(((Math.random() * 0.4) - 0.2).toFixed(4))
                }
            }
            newData.push(row)
        }
        return newData
    }

    const handleGenerate = (type) => {
        setActivePreset(type)
        setData(generatePatternData(type))
    }

    const handlePredict = () => {
        const numericData = data.map(row => row.map(val => parseFloat(val)))
        onPredict(numericData)
    }

    const handleCellChange = (rowIndex, colIndex, value) => {
        const newData = [...data]
        newData[rowIndex][colIndex] = value
        setData(newData)
        setActivePreset('CUSTOM')
    }

    return (
        <div className="manual-input-container">
            <div className="input-glass-card">
                <div className="input-header">
                    <div>
                        <h2>Data Simulation Lab</h2>
                        <p>Simulate sensor data patterns or craft your own movement sequences</p>
                    </div>
                </div>

                <div className="preset-section">
                    <label className="section-label">Select Pattern Preset</label>
                    <div className="preset-grid">
                        <button
                            className={`preset-btn ${activePreset === 'WALKING' ? 'active' : ''}`}
                            onClick={() => handleGenerate('WALKING')}
                        >
                            Walking
                        </button>
                        <button
                            className={`preset-btn ${activePreset === 'WALKING_UPSTAIRS' ? 'active' : ''}`}
                            onClick={() => handleGenerate('WALKING_UPSTAIRS')}
                        >
                            Upstairs
                        </button>
                        <button
                            className={`preset-btn ${activePreset === 'WALKING_DOWNSTAIRS' ? 'active' : ''}`}
                            onClick={() => handleGenerate('WALKING_DOWNSTAIRS')}
                        >
                            Downstairs
                        </button>
                        <button
                            className={`preset-btn ${activePreset === 'SITTING' ? 'active' : ''}`}
                            onClick={() => handleGenerate('SITTING')}
                        >
                            Sitting
                        </button>
                        <button
                            className={`preset-btn ${activePreset === 'STANDING' ? 'active' : ''}`}
                            onClick={() => handleGenerate('STANDING')}
                        >
                            Standing
                        </button>
                        <button
                            className={`preset-btn ${activePreset === 'LAYING' ? 'active' : ''}`}
                            onClick={() => handleGenerate('LAYING')}
                        >
                            Laying
                        </button>
                        <button
                            className={`preset-btn random-btn ${activePreset === 'RANDOM' ? 'active' : ''}`}
                            onClick={() => handleGenerate('RANDOM')}
                        >
                            Random
                        </button>
                    </div>
                </div>

                <div className="table-controls">
                    <div className="rows-info">
                        Showing <strong>{rows}</strong> of 128 timesteps
                    </div>
                    <div className="view-selector">
                        <label>View Depth:</label>
                        <select value={rows} onChange={(e) => setRows(Number(e.target.value))}>
                            <option value={10}>Compact (10)</option>
                            <option value={20}>Standard (20)</option>
                            <option value={50}>Detailed (50)</option>
                            <option value={128}>Full Dataset (128)</option>
                        </select>
                    </div>
                </div>

                <div className="data-grid-wrapper">
                    <table className="data-grid">
                        <thead>
                            <tr>
                                <th className="sticky-col">#</th>
                                <th>Body Acc X</th>
                                <th>Body Acc Y</th>
                                <th>Body Acc Z</th>
                                <th>Gyro X</th>
                                <th>Gyro Y</th>
                                <th>Gyro Z</th>
                                <th>Total Acc X</th>
                                <th>Total Acc Y</th>
                                <th>Total Acc Z</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.slice(0, rows).map((row, i) => (
                                <tr key={i}>
                                    <td className="sticky-col row-label">{i + 1}</td>
                                    {row.map((val, j) => (
                                        <td key={j}>
                                            <input
                                                className="cell-input"
                                                type="number"
                                                step="0.0001"
                                                value={val}
                                                onChange={(e) => handleCellChange(i, j, e.target.value)}
                                            />
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {rows < 128 && (
                        <div className="scroll-indicator">
                            <div className="indicator-text">↓ {128 - rows} more timesteps hidden ↓</div>
                        </div>
                    )}
                </div>

                <div className="action-footer">
                    <button
                        className={`lab-predict-btn ${loading ? 'loading' : ''}`}
                        onClick={handlePredict}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <div className="loader-dots">
                                    <span></span><span></span><span></span>
                                </div>
                                Running Inference...
                            </>
                        ) : (
                            <>
                                Run Prediction Model
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
}

export default ManualInput
