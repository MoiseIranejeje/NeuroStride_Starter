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
    title: 'Small knee extensions',
    data: scan, summary: scanSummary, episodes: scanEpisodes,
    reading: 'At +0.04 rad, Go2 walked 0.75 cm taller on average and finished 1.53 cm closer to the centerline than baseline. All three runs passed, with worst peak yaw reaching 9.29°.',
    script: 'go2_knee_extension_probe.py',
  },
  development: {
    title: 'Neural upright controller',
    data: upright, summary: uprightSummary, episodes: uprightEpisodes,
    reading: 'The neural posture controller reached 39.49 cm and lowered mean peak yaw to 3.19°. Mean final sideways offset grew to 65.33 cm and all three runs failed the straight-walk criteria.',
    script: 'go2_snn_upright_walk.py',
  },
}

type StudyId = keyof typeof studies
type Row = Pick<(typeof scan)[number], 'mean_abs_final_y_m' | 'mean_visual_top_height_m' | 'mean_peak_yaw_deg'>

function modeLabel(mode: string) {
  if (mode === 'baseline') return 'Baseline'
  if (mode === 'neural') return 'Neural'
  return `+0.${mode.slice(5)} rad`
}

const metrics = {
  drift: { label: 'Final sideways offset', unit: 'cm', scale: 100, reference: 25, value: (row: Row) => row.mean_abs_final_y_m * 100 },
  height: { label: 'Visual top height', unit: 'cm', scale: 45, reference: 40, value: (row: Row) => row.mean_visual_top_height_m * 100 },
  yaw: { label: 'Peak yaw', unit: '°', scale: 10, reference: 10, value: (row: Row) => row.mean_peak_yaw_deg },
}

function App() {
  const [studyId, setStudyId] = useState<StudyId>('extension_scan')
  const [metricId, setMetricId] = useState<keyof typeof metrics>('drift')
  const study = studies[studyId]
  const metric = metrics[metricId]
  const reference = metric.reference / metric.scale * 100

  return (
    <>
      <a className="skip" href="#evidence">Skip to evidence</a>
      <header className="site-header">
        <div className="wrap flex items-center justify-between gap-4">
          <a href="#" className="wordmark"><span aria-hidden="true">∿</span> NeuroStride</a>
          <nav aria-label="Page sections" className="flex flex-wrap gap-x-5 gap-y-2">
            <a href="#evidence">Results</a><a href="#methods">Methods</a>
          </nav>
        </div>
      </header>
      <main className="wrap">
        <section className="hero grid gap-8 lg:grid-cols-[1fr_16rem] lg:items-end">
          <div>
            <p className="eyebrow">Iterate / Go2 locomotion</p>
            <h1>Walking taller.<br /><em>Drifting sideways.</em></h1>
            <p className="lede">In MuJoCo, Go2 reached nearly 40 cm with a neural posture controller. Its heading improved, but it drifted more than 65 cm sideways on average.</p>
          </div>
          <dl className="protocol">
            <div><dt>Environment</dt><dd>MuJoCo simulation</dd></div>
            <div><dt>Command</dt><dd>0.45 m/s · 8 s</dd></div>
            <div><dt>Sample</dt><dd>3 runs per mode</dd></div>
          </dl>
        </section>

        <section id="evidence" className="evidence" aria-labelledby="results-heading">
          <div className="flex flex-wrap items-end justify-between gap-5">
            <h2 id="results-heading">Experiment results</h2>
            <div className="flex flex-wrap gap-2" role="group" aria-label="Experiment">
              {(Object.keys(studies) as StudyId[]).map(id =>
                <button key={id} aria-pressed={studyId === id} onClick={() => setStudyId(id)}>{studies[id].title}</button>
              )}
            </div>
          </div>
          <p className="reading" aria-live="polite">{study.reading}</p>
          <div className="figure">
            <div className="flex flex-wrap justify-between items-center gap-4">
              <h3>Mean {metric.label.toLowerCase()}</h3>
              <div className="flex flex-wrap gap-2" role="group" aria-label="Chart metric">
                {(Object.keys(metrics) as (keyof typeof metrics)[]).map(id =>
                  <button key={id} aria-pressed={metricId === id} onClick={() => setMetricId(id)}>{metrics[id].label}</button>
                )}
              </div>
            </div>
            <ul className="bars" aria-label={`Mean ${metric.label.toLowerCase()} by mode`}>
              {study.data.map(row => (
                <li key={row.mode}>
                  <span>{modeLabel(row.mode)}</span>
                  <div className="track" aria-hidden="true">
                    <div className="fill" style={{ width: `${metric.value(row) / metric.scale * 100}%` }} />
                    <div className="reference" style={{ left: `${reference}%` }} />
                  </div>
                  <strong>{metric.value(row).toFixed(2)} {metric.unit}</strong>
                </li>
              ))}
            </ul>
            <p className="caption">Scale 0 to {metric.scale} {metric.unit}. Dashed marker at {metric.reference} {metric.unit} {metricId === 'height' ? 'is the height target.' : 'shows an individual-run limit. Pass counts below use individual runs and all task criteria.'}</p>
          </div>

          <div className="table-scroll" tabIndex={0} role="region" aria-label={`${study.title} results table`}>
            <table>
              <caption className="sr-only">{study.title} results by mode</caption>
              <thead><tr>{['Mode', 'Top height (cm)', 'Final offset (cm)', 'Peak cross-track (cm)', 'Mean peak yaw (°)', 'Worst peak yaw (°)', '10° task passes', '15° task passes'].map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
              <tbody>{study.data.map(row => (
                <tr key={row.mode}>
                  <th scope="row">{modeLabel(row.mode)}</th>
                  <td>{(row.mean_visual_top_height_m * 100).toFixed(2)}</td>
                  <td>{(row.mean_abs_final_y_m * 100).toFixed(2)}</td>
                  <td>{(row.mean_peak_cross_track_m * 100).toFixed(2)}</td>
                  <td>{row.mean_peak_yaw_deg.toFixed(2)}</td>
                  <td>{row.worst_peak_yaw_deg.toFixed(2)}</td>
                  <td><strong className={row.pass_10_count === row.n ? 'pass' : 'fail'}>{row.pass_10_count}/{row.n}</strong></td>
                  <td>{row.pass_15_count}/{row.n}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="downloads flex flex-wrap gap-x-6 gap-y-2">
            <a href={study.summary} download={studyId + '-summary.json'}>Download summary JSON ↓</a>
            <a href={study.episodes} download={studyId + '-episodes.csv'}>Download episode CSV ↓</a>
          </div>
          <p className="caption">Source script <code>{study.script}</code></p>
        </section>
        <section id="methods" className="methods grid gap-8 md:grid-cols-2" aria-label="Methods and limitations">
          <div>
            <h2>What counts as a pass?</h2>
            <p>Finish the full 8 seconds, travel at least 1.5 m forward, keep final sideways offset within 25 cm, and maintain a minimum root height of 18 cm. The two pass columns apply peak yaw limits of 10° and 15° respectively.</p>
            <p>Visual top height measures the top of the rendered robot after the first 0.75 s. Final offset measures where it finishes. Peak cross-track captures the greatest sideways deviation during the walk.</p>
          </div>
          <div>
            <h2>What remains open?</h2>
            <p>Both experiments use the same three development seeds (7601–7603) at one commanded speed. These runs support a follow-up on unseen seeds. Hardware performance and generalization to other speeds or terrain remain untested by these experiments.</p>
            <p>The Build (CartPole) story and episode videos are reserved for future releases.</p>
          </div>
        </section>
      </main>
      <footer className="wrap footer">NeuroStride / TKS Build &amp; Iterate</footer>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<App />)
