import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import scan from './evidence/extension_scan-summary.json'
import upright from './evidence/development-summary.json'
import scanSummary from './evidence/extension_scan-summary.json?url'
import scanEpisodes from './evidence/extension_scan-episodes.csv?url'
import uprightSummary from './evidence/development-summary.json?url'
import uprightEpisodes from './evidence/development-episodes.csv?url'
import scanEpisodeCsv from './evidence/extension_scan-episodes.csv?raw'
import uprightEpisodeCsv from './evidence/development-episodes.csv?raw'
import replayProvenance from '../public/media/go2-hero-provenance.json'
import { HeroVideo } from './HeroVideo'
import './styles.css'

const studies = {
  extension_scan: {
    title: 'Fixed knee extensions',
    label: 'STUDY A / STANCE OFFSETS',
    subtitle: 'Effect of fixed knee-extension offsets on the walking trajectory',
    reading: 'At +0.04 rad, mean rendered-mesh height increased from 35.86 to 36.61 cm and mean absolute final lateral displacement decreased from 9.42 to 7.89 cm (3/3 full-task passes). At +0.12 rad, mean lateral displacement reached 32.71 cm (0/3 passes).',
    data: scan,
    summary: scanSummary,
    episodes: scanEpisodes,
    script: 'go2_knee_extension_probe.py',
  },
  development: {
    title: 'Spiking height feedback',
    label: 'STUDY B / NEURAL FEEDBACK',
    subtitle: 'Effect of a spiking height-feedback controller on locomotion and path tracking',
    reading: 'A separate spiking controller adjusts leg extension from root-height error and vertical velocity. Trained by supervised imitation of an analytic feedback rule, it reached 39.49 cm mean visual height but 65.33 cm mean absolute final lateral displacement (0/3 full-task passes).',
    data: upright,
    summary: uprightSummary,
    episodes: uprightEpisodes,
    script: 'go2_snn_upright_walk.py',
  },
} as const

type StudyId = keyof typeof studies
type EvidenceRow = Pick<(typeof scan)[number],
  'mean_abs_final_y_m' | 'mean_visual_top_height_m' | 'mean_peak_yaw_deg'>

const metrics = {
  drift: {
    label: 'Final sideways offset',
    short: 'Lateral drift',
    unit: 'cm',
    scale: 100,
    reference: 25,
    value: (row: EvidenceRow) => row.mean_abs_final_y_m * 100,
    note: 'The dashed line marks the 25 cm final-offset limit for an individual run. Bars show averages; the pass counts use every criterion on individual runs.',
  },
  height: {
    label: 'Rendered visual height',
    short: 'Visual height',
    unit: 'cm',
    scale: 45,
    reference: 40,
    value: (row: EvidenceRow) => row.mean_visual_top_height_m * 100,
    note: 'The 40 cm line is the manufacturer’s nominal overall standing height. These bars measure the top of the rendered MuJoCo mesh, not the torso origin or physical hardware.',
  },
  yaw: {
    label: 'Mean peak yaw',
    short: 'Heading error',
    unit: '°',
    scale: 12,
    reference: 10,
    value: (row: EvidenceRow) => row.mean_peak_yaw_deg,
    note: 'The dashed line marks the 10° maximum yaw limit for an individual run. Low yaw means the robot is oriented forward; it does not establish low sideways drift.',
  },
} as const

type MetricId = keyof typeof metrics

const fmtCm = (metres: number) => (metres * 100).toFixed(2)
const fmtDeg = (degrees: number) => degrees.toFixed(2)

function modeLabel(mode: string) {
  if (mode === 'baseline') return 'Original stance'
  if (mode === 'neural') return 'Posture SNN'
  const number = Number(mode.slice('fixed'.length)) / 100
  return '+' + number.toFixed(2) + ' rad'
}

const original = upright.find(row => row.mode === 'baseline')!
const smallOffset = scan.find(row => row.mode === 'fixed04')!
const neural = upright.find(row => row.mode === 'neural')!

// The simulator exports unquoted scalar fields and relative replay paths.
function readEpisodes(csv: string) {
  const [header, ...lines] = csv.trim().split(/\r?\n/)
  const fields = header.split(',')
  return lines.map(line => {
    const values = line.split(',')
    return Object.fromEntries(fields.map((field, index) => [field, values[index]]))
  })
}

const episodeRows = {
  extension_scan: readEpisodes(scanEpisodeCsv),
  development: readEpisodes(uprightEpisodeCsv),
}

function EpisodeExplorer() {
  const [studyId, setStudyId] = useState<StudyId>('development')
  const [seed, setSeed] = useState('')
  const [outcome, setOutcome] = useState('')
  const rows = episodeRows[studyId]
  const seeds = [...new Set(rows.map(row => row.seed))]
  const visible = rows.filter(row =>
    (!seed || row.seed === seed) && (!outcome || row.pass_10deg === outcome),
  )

  return (
    <section className="episodes-section" id="episodes" aria-labelledby="episodes-title">
      <div className="container">
        <p className="eyebrow section-index">04 — SIMULATION RECORDINGS</p>
        <h2 id="episodes-title">Episode explorer</h2>
        <p>Inspect individual MuJoCo runs. Pass status uses all task criteria, including the 10° heading limit and the 25 cm final sideways limit.</p>
        <div className="episode-filters">
          <label>Study
            <select value={studyId} onChange={event => setStudyId(event.target.value as StudyId)}>
              <option value="development">{studies.development.title}</option>
              <option value="extension_scan">{studies.extension_scan.title}</option>
            </select>
          </label>
          <label>Seed
            <select value={seed} onChange={event => setSeed(event.target.value)}>
              <option value="">All seeds</option>
              {seeds.map(value => <option key={value}>{value}</option>)}
            </select>
          </label>
          <label>Outcome
            <select value={outcome} onChange={event => setOutcome(event.target.value)}>
              <option value="">All outcomes</option>
              <option value="True">Passed</option>
              <option value="False">Failed</option>
            </select>
          </label>
        </div>
        <p role="status">{visible.length} of {rows.length} runs shown</p>
        {visible.length > 0 && (
          <div className="table-scroll episode-table" tabIndex={0} role="region" aria-label="Episode records">
            <table>
              <thead><tr>
                <th scope="col">Controller</th><th scope="col">Seed</th><th scope="col">Duration</th>
                <th scope="col">Final sideways offset</th><th scope="col">Peak heading error</th>
                <th scope="col">Outcome</th><th scope="col">Replay</th>
              </tr></thead>
              <tbody>{visible.map(row => {
                const hasReplay = studyId === 'development'
                  && row.replay === `replays/${replayProvenance.source_filename}`
                return (
                  <tr key={`${studyId}/${row.mode}/${row.seed}/${row.command_mps}`}>
                    <th scope="row">{modeLabel(row.mode)}</th><td>{row.seed}</td><td>{row.duration_s} s</td>
                    <td>{fmtCm(Math.abs(Number(row.sideways_m)))} cm</td><td>{fmtDeg(Number(row.max_abs_yaw_deg))}°</td>
                    <td>{row.pass_10deg === 'True' ? 'Passed' : 'Failed'}</td>
                    <td>{hasReplay ? <a href="#episode-replay">Watch replay</a> : 'Unavailable'}</td>
                  </tr>
                )
              })}</tbody>
            </table>
          </div>
        )}
        <a className="episode-download" href={studies[studyId].episodes} download>Download study CSV</a>
        <div className="episode-replay" id="episode-replay">
          <h3>Neural height feedback · seed 7601</h3>
          <p>Eight-second replay rendered from saved simulator states. The Go2 ended 65.97 cm sideways.</p>
          <video controls playsInline preload="none" aria-label="MuJoCo replay of the neural height-feedback controller, seed 7601">
            <source src="/media/go2-hero.webm" type="video/webm" />
            <source src="/media/go2-hero.mp4" type="video/mp4" />
          </video>
          <a href="/media/go2-hero.mp4">Open replay MP4</a>
        </div>
      </div>
    </section>
  )
}

function App() {
  const [studyId, setStudyId] = useState<StudyId>('extension_scan')
  const [metricId, setMetricId] = useState<MetricId>('drift')
  const study = studies[studyId]
  const metric = metrics[metricId]
  const lineAt = metric.reference / metric.scale * 100

  return (
    <>
      <a className="skip-link" href="#results">Skip to research results</a>
      <header className="site-header">
        <div className="container header-inner">
          <a href="#top" className="brand" aria-label="NeuroStride homepage">
            <span className="brand-symbol" aria-hidden="true">∿</span>
            <span>NEURO<span className="brand-weight">STRIDE</span></span>
          </a>
          <nav aria-label="Site navigation" className="site-nav">
            <a href="#story">Iterate</a>
            <a href="#results">Results</a>
            <a href="#methods">Protocol</a>
            <a href="#episodes">Episodes</a>
          </nav>
          <a className="header-action" href="#results">View results <span aria-hidden="true">↗</span></a>
        </div>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-heading">
          <div className="container hero-inner">
            <div className="hero-copy">
              <p className="eyebrow hero-eyebrow"><span className="signal-dot" /> NEUROSTRIDE / MUJOCO GO2 / SNN CONTROL</p>
              <h1 id="hero-heading">Spiking neural control<br /><em>of Unitree Go2.</em></h1>
              <p className="hero-description">We investigate whether spiking neural network (SNN) controllers can maintain forward locomotion, heading, and lateral path accuracy as the Go2 leg posture changes. The experiments compare fixed knee-extension offsets with a separately trained spiking height-feedback controller in MuJoCo.</p>
              <div className="hero-actions">
                <a className="primary-link" href="#story">Explore the iterations <span aria-hidden="true">↘</span></a>
              </div>
            </div>
            <div className="hero-visuals">
              <HeroVideo />
              <aside className="hero-evidence" aria-label="Neural upright walking outcomes">
                <div className="hero-evidence-top"><span>HEIGHT-FEEDBACK SNN RESULTS</span><span>SEEDS 7601–7603</span></div>
                <div className="hero-metric">
                  <span className="hero-metric-index">VISUAL HEIGHT</span>
                  <strong>{fmtCm(neural.mean_visual_top_height_m)}<small>cm</small></strong>
                  <span>Mean rendered-mesh top height after 0.75 s</span>
                </div>
                <div className="hero-metric">
                  <span className="hero-metric-index">HEADING ERROR</span>
                  <strong>{fmtDeg(neural.mean_peak_yaw_deg)}<small>°</small></strong>
                  <span>Mean per-episode peak yaw</span>
                </div>
                <div className="hero-metric hero-metric-warning">
                  <span className="hero-metric-index">LATERAL DISPLACEMENT</span>
                  <strong>{fmtCm(neural.mean_abs_final_y_m)}<small>cm</small></strong>
                  <span>Mean absolute final Y; limit is 25 cm per run</span>
                </div>
              </aside>
            </div>
          </div>
        </section>

        <section className="story-section container section" id="story" aria-labelledby="story-title">
          <div className="section-intro">
            <p className="eyebrow section-index">01 — ITERATE</p>
            <h2 id="story-title">How does leg extension<br />affect SNN locomotion?</h2>
            <p className="section-lede">Can the Go2 walk taller and still stay on course? We kept the gait and heading networks fixed and changed how much the knees extend. Each configuration ran on the same three MuJoCo development seeds.</p>
          </div>
          <div className="story-grid">
            <article className="story-card">
              <span className="card-step">MEAN FINAL SIDEWAYS DRIFT</span>
              <strong className="card-number">{fmtCm(original.mean_abs_final_y_m)}<small> cm</small></strong>
              <h3>Baseline gait controller</h3>
              <p>The original gait and heading networks stayed fixed, with zero added knee extension. The number shows how far sideways the robot ended up, averaged across the three runs.</p>
              <span className="card-outcome positive">3 of 3 passed all criteria (10° heading limit)</span>
            </article>
            <article className="story-card">
              <span className="card-step">MEAN FINAL SIDEWAYS DRIFT</span>
              <strong className="card-number">{fmtCm(smallOffset.mean_abs_final_y_m)}<small> cm</small></strong>
              <h3>Test fixed knee offsets</h3>
              <p>We used fixed offsets to see how a taller stance affected walking. At +0.04 rad, the Go2 stood slightly taller and passed all three runs. At +0.12 rad, all three runs failed.</p>
              <a className="card-outcome story-link" href="#results" onClick={() => {
                setStudyId('extension_scan')
                setMetricId('drift')
              }}>Compare fixed offsets <span aria-hidden="true">↘</span></a>
            </article>
            <article className="story-card story-card-critical">
              <span className="card-step">MEAN TOP-OF-ROBOT HEIGHT</span>
              <strong className="card-number">{fmtCm(neural.mean_visual_top_height_m)}<small> cm</small></strong>
              <h3>Try neural height feedback</h3>
              <p>A separate spiking network adjusted knee extension from body-height error and vertical speed. Mean height rose from {fmtCm(original.mean_visual_top_height_m)} cm and peak heading error fell on average. Mean final sideways drift reached {fmtCm(neural.mean_abs_final_y_m)} cm, and every run exceeded the 25 cm limit.</p>
              <span className="card-outcome negative">0 of 3 passed all criteria</span>
              <a className="story-link" href="#episode-replay">See the seed 7601 replay <span aria-hidden="true">↘</span></a>
            </article>
          </div>
          <div className="insight-line"><span aria-hidden="true">↳</span><p><strong>Next experiment.</strong> Adapt the gait controller to the taller stance, then test on fresh seeds to see whether the Go2 can maintain its height and stay within the heading and sideways-drift limits.</p></div>
        </section>

        <section className="lab-section" id="results" aria-labelledby="results-title">
          <div className="container">
            <div className="lab-heading">
              <div>
                <p className="eyebrow section-index">02 — CONTROLLER COMPARISONS</p>
                <h2 id="results-title">Experimental results</h2>
                <p>Compare recorded MuJoCo evaluation results from seeds 7601–7603. Select a controller study and measurement; the CSV downloads contain the individual episode data.</p>
              </div>
            </div>

            <div className="study-switch" role="group" aria-label="Choose an experiment">
              {(Object.keys(studies) as StudyId[]).map(id => (
                <button type="button" key={id} className={'study-tab' + (studyId === id ? ' active' : '')}
                  aria-pressed={studyId === id} onClick={() => setStudyId(id)}>
                  <span>{studies[id].label}</span><strong>{studies[id].title}</strong>
                </button>
              ))}
            </div>

            <div className="analysis-panel">
              <div className="analysis-header">
                <div className="analysis-copy">
                  <p className="eyebrow">MUJOCO EVALUATION / {study.data[0].n} DEVELOPMENT STARTS PER CONDITION</p>
                  <h3>{study.subtitle}</h3>
                  <p aria-live="polite">{study.reading}</p>
                </div>
                <div className="analysis-protocol">
                  <span>Test protocol</span>
                  <strong>8 s · flat ground</strong>
                  <span>Commanded 0.45 m/s</span>
                  <span>Seeds 7601–7603</span>
                </div>
              </div>

              <div className="chart-heading">
                <div>
                  <p className="eyebrow">SELECT PLOTTED METRIC</p>
                  <h4>{metric.label}</h4>
                </div>
                <div className="metric-switch" role="group" aria-label="Choose chart measurement">
                  {(Object.keys(metrics) as MetricId[]).map(id => (
                    <button type="button" key={id} className={metricId === id ? 'active' : ''}
                      aria-pressed={metricId === id} onClick={() => setMetricId(id)}>{metrics[id].short}</button>
                  ))}
                </div>
              </div>

              <div className="chart" role="group" aria-label={'Comparison of ' + metric.label.toLowerCase()}>
                <div className="chart-scale" aria-hidden="true"><span>0 {metric.unit}</span><span>{(metric.scale / 2).toFixed(0)} {metric.unit}</span><span>{metric.scale} {metric.unit}</span></div>
                <ul className="chart-bars">
                  {study.data.map(row => (
                    <li className="chart-row" key={row.mode}>
                      <span className="chart-label">{modeLabel(row.mode)}</span>
                      <div className="chart-track" aria-hidden="true">
                        <div className={'chart-fill' + (row.pass_10_count === row.n ? ' success' : '')}
                          style={{ width: Math.min(100, metric.value(row) / metric.scale * 100) + '%' }} />
                        <div className="chart-target" style={{ insetInlineStart: lineAt + '%' }} />
                      </div>
                      <strong className="chart-value">{metric.value(row).toFixed(2)}<small> {metric.unit}</small></strong>
                    </li>
                  ))}
                </ul>
                <p className="chart-note"><span className="marker-key" aria-hidden="true" />{metric.note}</p>
              </div>

              <div className="data-bottom">
                <div className="data-qualifier"><strong>Full-task pass:</strong> one episode must meet the duration, forward-distance, lateral-position, minimum-root-height and peak-yaw limits together.</div>
                <div className="data-exports">
                  <a href={study.summary} download={studyId + '-summary.json'}>Summary JSON <span aria-hidden="true">↗</span></a>
                  <a href={study.episodes} download={studyId + '-episodes.csv'}>Episode CSV <span aria-hidden="true">↗</span></a>
                </div>
              </div>
            </div>

            <details className="results-details">
              <summary><span><strong>Inspect the full measurement table</strong><small>All configurations, measured heights, yaw, deviation and pass counts</small></span><span className="expand-symbol" aria-hidden="true">+</span></summary>
              <div className="table-scroll" tabIndex={0} role="region" aria-label={study.title + ' measurements'}>
                <table>
                  <caption className="sr-only">{study.title} — observed mean metrics on three development starts</caption>
                  <thead><tr>
                    {['Mode', 'Top height (cm)', 'Final offset (cm)', 'Peak path deviation (cm)', 'Mean peak yaw (°)', 'Worst peak yaw (°)', '10° full-task passes', '15° full-task passes'].map(t => <th key={t} scope="col">{t}</th>)}
                  </tr></thead>
                  <tbody>{study.data.map(row => (
                    <tr key={row.mode}>
                      <th scope="row">{modeLabel(row.mode)}</th>
                      <td>{fmtCm(row.mean_visual_top_height_m)}</td>
                      <td>{fmtCm(row.mean_abs_final_y_m)}</td>
                      <td>{fmtCm(row.mean_peak_cross_track_m)}</td>
                      <td>{fmtDeg(row.mean_peak_yaw_deg)}</td>
                      <td>{fmtDeg(row.worst_peak_yaw_deg)}</td>
                      <td><span className={row.pass_10_count === row.n ? 'table-pass' : 'table-fail'}>{row.pass_10_count}/{row.n}</span></td>
                      <td>{row.pass_15_count}/{row.n}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              <p className="data-caption">Source script: {study.script}. Data represent three development seeds, not a statistically validated hardware benchmark.</p>
            </details>
          </div>
        </section>

        <section className="method-section container section" id="methods" aria-labelledby="methods-title">
          <div className="method-intro">
            <p className="eyebrow section-index">03 — METHODOLOGY</p>
            <h2 id="methods-title">Simulation and<br />evaluation protocol</h2>
          </div>
          <div className="method-grid">
            <article>
              <span className="method-number">01</span>
              <h3>Controller architecture</h3>
              <p>The existing gait SNN outputs 12 joint actions; a separate yaw SNN supplies heading correction. Both remain frozen. The additional height SNN was trained by supervised imitation of an analytic feedback controller, not end-to-end reinforcement learning. Motor torques use a conventional PD controller.</p>
            </article>
            <article>
              <span className="method-number">02</span>
              <h3>Full-task success criteria</h3>
              <p>An episode must complete 8 s, move at least 1.5 m forward, end within ±0.25 m laterally and maintain root height ≥0.18 m. Two success columns use per-episode peak yaw thresholds of 10° and 15°. All criteria must hold for a pass.</p>
            </article>
            <article>
              <span className="method-number">03</span>
              <h3>Measurement limitations</h3>
              <p>Visual height is the highest rendered-mesh point after the first 0.75 s, not the torso-origin height. Only three development starts at 0.45 m/s on flat terrain were examined. Generalization, foot-contact mechanisms and real hardware have not been established.</p>
            </article>
          </div>
        </section>

        <EpisodeExplorer />
      </main>

      <footer className="site-footer">
        <div className="container footer-inner">
          <span className="footer-brand">∿ NeuroStride</span>
          <p>Spiking neural control experiments using Unitree Go2 in MuJoCo. Simulation results only.</p>
          <a href="#top">Back to top ↑</a>
        </div>
      </footer>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
