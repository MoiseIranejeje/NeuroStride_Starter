import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import scan from './evidence/extension_scan-summary.json'
import upright from './evidence/development-summary.json'
import scanSummary from './evidence/extension_scan-summary.json?url'
import scanEpisodes from './evidence/extension_scan-episodes.csv?url'
import uprightSummary from './evidence/development-summary.json?url'
import uprightEpisodes from './evidence/development-episodes.csv?url'
import './styles.css'

const studies = {
  extension_scan: {
    title: 'Small extensions',
    label: 'Experiment 02',
    subtitle: 'How far can we extend the legs before path tracking breaks?',
    reading: 'The +0.04 rad stance increased visual height and slightly reduced lateral error on three development starts. At +0.12 rad, all three runs failed the full straight-walking test.',
    data: scan,
    summary: scanSummary,
    episodes: scanEpisodes,
    script: 'go2_knee_extension_probe.py',
  },
  development: {
    title: 'Upright SNN',
    label: 'Experiment 01',
    subtitle: 'A third spiking controller regulates leg extension in real time.',
    reading: 'The new posture SNN held the rendered robot near 40 cm and improved yaw, but increased average final lateral drift to 65.33 cm. None of its three runs passed.',
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
const neural = upright.find(row => row.mode === 'neural')!
const modest = scan.find(row => row.mode === 'fixed04')!

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
            <a href="#story">The story</a>
            <a href="#results">Evidence</a>
            <a href="#methods">Method</a>
            <a href="#episodes">Episodes</a>
          </nav>
          <a className="header-action" href="#results">Explore the data <span aria-hidden="true">↗</span></a>
        </div>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-heading">
          <div className="container hero-inner">
            <div className="hero-copy">
              <p className="eyebrow hero-eyebrow"><span className="signal-dot" /> NEUROSTRIDE / RESEARCH NOTE 001</p>
              <h1 id="hero-heading">Standing taller.<br /><em>Missing the line.</em></h1>
              <p className="hero-description">We taught a simulated Unitree Go2 to walk with spiking neural networks. Making it stand taller seemed like progress—until we measured where it actually ended up.</p>
              <div className="hero-actions">
                <a className="primary-link" href="#story">Read the discovery <span aria-hidden="true">↘</span></a>
                <span className="hero-meta">MuJoCo · 0.45 m/s command · 8 s runs</span>
              </div>
            </div>
            <aside className="hero-evidence" aria-label="Neural upright walking outcomes">
              <div className="hero-evidence-top"><span>THE EXPERIMENT IN THREE NUMBERS</span><span>GO2 / ITERATE</span></div>
              <div className="hero-metric">
                <span className="hero-metric-index">01 / POSTURE</span>
                <strong>{fmtCm(neural.mean_visual_top_height_m)}<small>cm</small></strong>
                <span>Mean rendered top height, neural mode</span>
              </div>
              <div className="hero-metric">
                <span className="hero-metric-index">02 / HEADING</span>
                <strong>{fmtDeg(neural.mean_peak_yaw_deg)}<small>°</small></strong>
                <span>Mean peak yaw across three starts</span>
              </div>
              <div className="hero-metric hero-metric-warning">
                <span className="hero-metric-index">03 / PATH</span>
                <strong>{fmtCm(neural.mean_abs_final_y_m)}<small>cm</small></strong>
                <span>Mean final sideways offset · failed the 25 cm limit</span>
              </div>
              <div className="hero-evidence-bottom">Measured simulation results. No real-robot performance is claimed.</div>
            </aside>
          </div>
        </section>

        <section className="story-section container section" id="story" aria-labelledby="story-title">
          <div className="section-intro">
            <p className="eyebrow section-index">01 — THE FINDING</p>
            <h2 id="story-title">A straighter heading<br />isn’t a straighter path.</h2>
            <p className="section-lede">The posture controller did what it was trained to do: adjust leg extension using height and vertical-velocity feedback. But the walking network was learned for a different stance. Our measurements exposed the trade-off.</p>
          </div>
          <div className="story-grid">
            <article className="story-card">
              <span className="card-step">BEFORE / FROZEN WALKING SNN</span>
              <strong className="card-number">{fmtCm(original.mean_abs_final_y_m)}<small> cm</small></strong>
              <h3>The route held.</h3>
              <p>Original gait and yaw networks, no added knee extension. All {original.pass_10_count} of {original.n} development runs passed the full straight-walk criteria.</p>
              <span className="card-outcome positive">3 of 3 passed</span>
            </article>
            <article className="story-card">
              <span className="card-step">AFTER / NEW POSTURE SNN</span>
              <strong className="card-number">{fmtCm(neural.mean_visual_top_height_m)}<small> cm</small></strong>
              <h3>The body rose.</h3>
              <p>Height feedback was supplied by a new spiking module. The robot finished every eight-second episode, with a much taller rendered posture.</p>
              <span className="card-outcome neutral">Height improved</span>
            </article>
            <article className="story-card story-card-critical">
              <span className="card-step">THE UNEXPECTED RESULT</span>
              <strong className="card-number">{fmtCm(neural.mean_abs_final_y_m)}<small> cm</small></strong>
              <h3>The route slipped.</h3>
              <p>Despite low heading error, the taller neural configuration finished far outside the 25 cm lateral limit in all three tests.</p>
              <span className="card-outcome negative">0 of 3 passed</span>
            </article>
          </div>
          <div className="insight-line"><span aria-hidden="true">↳</span><p><strong>The lesson:</strong> orientation, body height and route accuracy are separate measurements. Optimizing one does not guarantee the others.</p></div>
        </section>

        <section className="lab-section" id="results" aria-labelledby="results-title">
          <div className="container">
            <div className="lab-heading">
              <div>
                <p className="eyebrow section-index">02 — EXPERIMENT DESK</p>
                <h2 id="results-title">See the evidence.</h2>
                <p>Every number below comes from stored MuJoCo experiment summaries. Change the study or measurement to investigate the trade-off.</p>
              </div>
              <div className="lab-method-tag"><span className="signal-dot" /> REAL SIMULATION DATA<br /><small>Not a scripted animation</small></div>
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
                  <p className="eyebrow">COMPARATIVE STUDY / {study.data[0].n} STARTS PER CONFIGURATION</p>
                  <h3>{study.subtitle}</h3>
                  <p aria-live="polite">{study.reading}</p>
                </div>
                <div className="analysis-protocol">
                  <span>Test conditions</span>
                  <strong>8 seconds / flat</strong>
                  <span>Commanded 0.45 m/s</span>
                  <span>Seeds 7601–7603</span>
                </div>
              </div>

              <div className="chart-heading">
                <div>
                  <p className="eyebrow">SELECT A MEASUREMENT</p>
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
                <div className="data-qualifier"><strong>Honest comparison.</strong> A configuration only passes if an individual run meets duration, forward, lateral, height and yaw conditions together.</div>
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
            <p className="eyebrow section-index">03 — HOW WE TESTED</p>
            <h2 id="methods-title">No perfect runs.<br />No hidden failures.</h2>
          </div>
          <div className="method-grid">
            <article>
              <span className="method-number">01</span>
              <h3>Frozen walking networks</h3>
              <p>All four conditions in each study share the existing gait SNN and spiking yaw controller. The upright study also tests a separately trained height-feedback SNN, while the fixed modes use conventional leg offsets.</p>
            </article>
            <article>
              <span className="method-number">02</span>
              <h3>One explicit pass rule</h3>
              <p>Each run must last 8 s, move at least 1.5 m forward, finish within 25 cm laterally and keep root height at least 18 cm. The heading criteria are reported separately at 10° and 15° peak yaw.</p>
            </article>
            <article>
              <span className="method-number">03</span>
              <h3>What remains unproven</h3>
              <p>These studies use three already examined starts, one flat terrain and one commanded speed. A visual mesh height near 40 cm is not proof of physical robot performance or reliable unseen-start navigation.</p>
            </article>
          </div>
          <div className="method-callout"><span className="eyebrow">NEXT QUESTION</span><p>Can a stance-adapted spiking gait preserve low lateral drift while keeping Go2 close to its natural 40 cm visual height?</p></div>
        </section>

        <section className="episodes-section" id="episodes" aria-labelledby="episodes-title">
          <div className="container episodes-inner">
            <div>
              <p className="eyebrow section-index">04 — EPISODE ARCHIVE</p>
              <h2 id="episodes-title">The runs, not a rendering.</h2>
              <p>Recorded MuJoCo episodes will live here alongside their actual seed, controller and outcome. We will publish real MP4 captures—not preprogrammed robot motion or illustrative footage presented as evidence.</p>
            </div>
            <div className="archive-card">
              <span className="archive-icon" aria-hidden="true">▶</span>
              <div><strong>Video episodes are being prepared.</strong><p>No verified MP4 footage has been added to this public archive yet. The current evidence can already be inspected and downloaded above.</p></div>
              <a href="#results">Explore the measurements <span aria-hidden="true">↗</span></a>
            </div>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="container footer-inner">
          <span className="footer-brand">∿ NeuroStride</span>
          <p>An open research record of simulated spiking locomotion. Built to show what worked—and what did not.</p>
          <a href="#top">Back to top ↑</a>
        </div>
      </footer>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
