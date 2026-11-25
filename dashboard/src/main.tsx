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
import episodeVideos from './evidence/episode-videos.json'
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
    referenceLabel: '25 cm run limit',
    value: (row: EvidenceRow) => row.mean_abs_final_y_m * 100,
    note: 'The dashed line marks the 25 cm limit. The shaded area shows offsets beyond the limit.',
  },
  height: {
    label: 'Rendered visual height',
    short: 'Visual height',
    unit: 'cm',
    referenceLabel: '40 cm nominal height',
    value: (row: EvidenceRow) => row.mean_visual_top_height_m * 100,
    note: 'The 40 cm line is the manufacturer’s nominal overall standing height. Each run measures the average height of the top of the rendered robot after the first 0.75 seconds.',
  },
  yaw: {
    label: 'Mean peak heading error',
    short: 'Heading error',
    unit: '°',
    referenceLabel: '10° run limit',
    value: (row: EvidenceRow) => row.mean_peak_yaw_deg,
    note: 'The dashed line marks the 10° run limit. The plot uses the largest absolute heading error measured in each run.',
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

const recordings: Record<string, { path: string }> = episodeVideos
const featuredReplay = 'development/replays/neural_v0.45_seed7601.npz'

function replayKey(studyId: StudyId, row: Record<string, string>) {
  return studyId + '/' + row.replay
}

function EpisodeExplorer({ replay, setReplay }: { replay: string; setReplay: (key: string) => void }) {
  const [studyId, setStudyId] = useState<StudyId>('development')
  const [seed, setSeed] = useState('')
  const [outcome, setOutcome] = useState('')
  const activeStudy: StudyId = replay.startsWith('development/') ? 'development' : 'extension_scan'
  const active = episodeRows[activeStudy].find(row => replayKey(activeStudy, row) === replay)
  const videoUrl = recordings[replay] && '/media/' + recordings[replay].path
  const activeLabel = active && (active.mode === 'neural' ? 'Spiking height feedback' : modeLabel(active.mode))
  const rows = episodeRows[studyId]
  const seeds = [...new Set(rows.map(row => row.seed))]
  const visible = rows.filter(row =>
    (!seed || row.seed === seed) && (!outcome || row.pass_10deg === outcome),
  )

  return (
    <section className="episodes-section" id="episodes" aria-labelledby="episodes-title">
      <div className="container">
        <p className="eyebrow section-index">05 — SIMULATION RECORDINGS</p>
        <h2 id="episodes-title">Episode explorer</h2>
        <p>Choose Watch replay beside a run to load its video player. Pass status uses all task criteria, including the 10° heading limit and the 25 cm final sideways limit.</p>
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
                const key = replayKey(studyId, row)
                return (
                  <tr key={`${studyId}/${row.mode}/${row.seed}/${row.command_mps}`}>
                    <th scope="row">{modeLabel(row.mode)}</th><td>{row.seed}</td><td>{row.duration_s} s</td>
                    <td>{fmtCm(Math.abs(Number(row.sideways_m)))} cm</td><td>{fmtDeg(Number(row.max_abs_yaw_deg))}°</td>
                    <td>{row.pass_10deg === 'True' ? 'Passed' : 'Failed'}</td>
                    <td>{recordings[key] ? (
                      <a href="#episode-replay" onClick={() => setReplay(key)}
                        aria-label={`Watch replay for ${modeLabel(row.mode)}, seed ${row.seed}`}>Watch replay</a>
                    ) : 'Unavailable'}</td>
                  </tr>
                )
              })}</tbody>
            </table>
          </div>
        )}
        <a className="episode-download" href={studies[studyId].episodes} download>Download study CSV</a>
        {active && videoUrl && (
          <div className="episode-replay" id="episode-replay">
            <h3>{activeLabel} · seed {active.seed}</h3>
            <p>{studies[activeStudy].title} · {active.duration_s} s · {fmtCm(Math.abs(Number(active.sideways_m)))} cm sideways · {active.pass_10deg === 'True' ? 'Passed' : 'Failed'}</p>
            <video key={videoUrl} controls playsInline preload="metadata" src={videoUrl}
              aria-label={`MuJoCo replay for ${activeLabel}, seed ${active.seed}`} />
            <a href={videoUrl}>Open replay MP4</a>
          </div>
        )}
      </div>
    </section>
  )
}

const runPlotMetrics = {
  height: { start: 34, end: 43, benchmark: 40, field: 'mean_visual_top_height_after_075s_m', scale: 100 },
  yaw: { start: 0, end: 12, benchmark: 10, field: 'max_abs_yaw_deg', scale: 1 },
  drift: { start: 0, end: 100, benchmark: 25, field: 'sideways_m', scale: 100 },
} as const

const comparisonSeeds = ['7601', '7602', '7603']

function SeedSelector({ seed, onChange }: { seed: string; onChange: (seed: string) => void }) {
  return (
    <label className="seed-selector">
      Highlight a run
      <select value={seed} onChange={event => onChange(event.target.value)}>
        <option value="">All three seeds</option>
        {comparisonSeeds.map(value => <option key={value} value={value}>Seed {value}</option>)}
      </select>
    </label>
  )
}

function RunMap({ studyId, metricId, modes, seed }: {
  studyId: StudyId
  metricId: MetricId
  modes: readonly string[]
  seed: string
}) {
  const metric = metrics[metricId]
  const plot = runPlotMetrics[metricId]
  const position = (value: number) => ((value - plot.start) / (plot.end - plot.start)) * 100 + '%'
  const valueOf = (row: Record<string, string>) => Math.abs(Number(row[plot.field])) * plot.scale
  const display = (value: number) => value.toFixed(2) + (metric.unit === '°' ? '°' : ' ' + metric.unit)
  return (
    <div className="run-map" role="group" aria-label={metric.label + ' by controller and seed'}>
      <div className="run-map-axis">
        <span>{plot.start}{metric.unit === '°' ? '°' : ' ' + metric.unit}</span>
        <span>{plot.end}{metric.unit === '°' ? '°' : ' ' + metric.unit}</span>
      </div>
      {modes.map(mode => {
        const runs = episodeRows[studyId].filter(run => run.mode === mode)
        const highlighted = runs.find(run => run.seed === seed)
        const mean = runs.reduce((sum, run) => sum + valueOf(run), 0) / runs.length
        return (
          <div className="run-map-row" data-neural={mode === 'neural'} key={mode}>
            <div className="run-map-label">
              <strong>{mode === 'baseline' ? 'Original controller' : mode === 'neural' ? 'Height-feedback SNN' : modeLabel(mode)}</strong>
              <span>{highlighted ? display(valueOf(highlighted)) : display(mean) + ' avg'}</span>
            </div>
            <div className="run-map-track" aria-hidden="true">
              {metricId !== 'height' && <div className="run-map-over" style={{ insetInlineStart: position(plot.benchmark) }} />}
              <div className="run-map-benchmark" style={{ insetInlineStart: position(plot.benchmark) }} />
              {runs.map(run => (
                <span key={run.seed} className="run-map-mark"
                  data-muted={Boolean(seed && seed !== run.seed)} data-selected={seed === run.seed}
                  style={{ insetInlineStart: position(valueOf(run)), top: (comparisonSeeds.indexOf(run.seed) + 1) * 25 + '%' }} />
              ))}
            </div>
            <span className="sr-only">{runs.map(run => 'Seed ' + run.seed + ': ' + display(valueOf(run))).join('. ')}</span>
          </div>
        )
      })}
      <p className="run-map-key"><span aria-hidden="true" />{metric.referenceLabel}</p>
    </div>
  )
}

function App() {
  const [studyId, setStudyId] = useState<StudyId>('extension_scan')
  const [replay, setReplay] = useState(featuredReplay)
  const [highlightedSeed, setHighlightedSeed] = useState('')
  const [metricId, setMetricId] = useState<MetricId>('drift')
  const study = studies[studyId]
  const metric = metrics[metricId]

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
            <a href="#snn">SNN</a>
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
              <a className="story-link" href="#episode-replay" onClick={() => setReplay(featuredReplay)}>See the seed 7601 replay <span aria-hidden="true">↘</span></a>
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

              <div className="study-plot">
                <RunMap studyId={studyId} metricId={metricId} modes={study.data.map(row => row.mode)} seed={highlightedSeed} />
                <div className="study-plot-notes">
                  <SeedSelector seed={highlightedSeed} onChange={setHighlightedSeed} />
                  <p>{metric.note}</p>
                  <p>Each mark is one run. Pick a seed to show its exact value for every controller here and in the SNN comparison.</p>
                </div>
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

        <section className="container section snn-section" id="snn" aria-labelledby="snn-title">
          <h2 id="snn-title">Why use a spiking neural network?</h2>
          <div className="snn-explanation">
            <article>
              <h3>Why we tested it</h3>
              <p>The Go2 already used spiking networks for gait and heading. A separate height-feedback SNN let us test whether knee extension could adapt during walking while keeping those two controllers fixed. This experiment measured locomotion in MuJoCo. Energy use and hardware performance were outside its scope.</p>
            </article>
            <article>
              <h3>How we trained it</h3>
              <p>We used supervised imitation. A hand-written feedback rule supplied target knee extensions based on how far the body was from its target height and how fast it was moving vertically. Two leaky integrate-and-fire layers learned that mapping offline. AdamW minimized prediction error with a small penalty on spike rate. During evaluation, the trained SNN supplied the extension command.</p>
            </article>
          </div>
          <h3>How it performed</h3>
          <p>On the same three development seeds, the original controller passed {original.pass_10_count} of {original.n} runs and the height-feedback SNN passed {neural.pass_10_count} of {neural.n}. The SNN increased height and reduced heading error, but every SNN run exceeded the final sideways limit.</p>
          <h4 className="trace-heading">Inside one run · seed 7601</h4>
          <p>These traces compare the original controller and the height-feedback SNN from the same MuJoCo starting seed at 0.45 m/s. The charts below summarize all three seeds.</p>
          <div className="trace-grid">
            <figure>
              <img src="/media/go2-seed7601-trajectory.png" width="1008" height="576" loading="lazy" alt="Forward versus sideways displacement. The SNN path passes the 0.25 m sideways limit, while the original path stays inside it." />
              <figcaption><strong>Walking path.</strong> Read left to right as the robot moves forward. The dotted lines mark ±25 cm sideways. The original run ended 9.52 cm off course, while the SNN run ended 65.97 cm off course.</figcaption>
            </figure>
            <figure>
              <img src="/media/go2-seed7601-pitch.png" width="1008" height="576" loading="lazy" alt="Torso pitch in degrees over eight seconds for both controllers." />
              <figcaption><strong>Torso pitch.</strong> This measures forward and backward tilt over time. Repeated peaks show how the torso rocks during the gait.</figcaption>
            </figure>
            <figure>
              <img src="/media/go2-seed7601-roll.png" width="1008" height="576" loading="lazy" alt="Torso roll in degrees over eight seconds for both controllers." />
              <figcaption><strong>Torso roll.</strong> This measures side-to-side tilt over time. Values above and below zero show the torso leaning in opposite directions.</figcaption>
            </figure>
            <figure>
              <img src="/media/go2-seed7601-heading.png" width="1008" height="576" loading="lazy" alt="Heading error over time. The SNN stays closer to zero, and both runs stay within the 10-degree limit." />
              <figcaption><strong>Heading error.</strong> Zero means the robot faces its starting direction. The dotted lines mark ±10°. Both runs stayed within that limit, even though the SNN drifted farther sideways.</figcaption>
            </figure>
          </div>
          <h4 className="trace-heading">Across all three seeds</h4>
          <p>Follow one seed across all three metrics, or show all runs with the controller averages. Each row shows where the runs landed relative to the benchmark.</p>
          <SeedSelector seed={highlightedSeed} onChange={setHighlightedSeed} />
          <div className="snn-charts">
            {([
              ['height', 'Each run contributes its mean height of the top of the rendered robot after 0.75 s. The dotted line shows the 40 cm nominal standing height. The SNN was taller on all three seeds.'],
              ['yaw', 'Each run contributes its largest absolute heading error in an eight-second run. Both controllers stayed below the dashed 10° limit, and the SNN had less heading error on every seed.'],
              ['drift', 'Each run contributes its absolute distance sideways when the run ended. Every SNN run exceeded the dashed 25 cm limit, while every original-controller run stayed below it.'],
            ] as const).map(([id, explanation]) => {
              const metric = metrics[id]
              const baselineMean = metric.value(original).toFixed(2)
              const neuralMean = metric.value(neural).toFixed(2)
              return (
                <figure className="snn-chart" key={id}>
                  <h5>{metric.label}</h5>
                  <RunMap studyId="development" metricId={id} modes={['baseline', 'neural']} seed={highlightedSeed} />
                  <figcaption>
                    <p className="snn-means">Three-run averages · {baselineMean} {metric.unit} original · {neuralMean} {metric.unit} SNN</p>
                    <p>{explanation}</p>
                  </figcaption>
                </figure>
              )
            })}
          </div>
        </section>

        <section className="method-section container section" id="methods" aria-labelledby="methods-title">
          <div className="method-intro">
            <p className="eyebrow section-index">04 — METHODOLOGY</p>
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

        <EpisodeExplorer replay={replay} setReplay={setReplay} />
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
