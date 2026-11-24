"""
Streamlit GUI for the MLR Learning Agent (same agent as simple_mlr_agent.py, driven from a browser)

  streamlit run app.py

Tabs follow the learning-agent components:
  Data              -> percepts before modeling
  Critic            -> MLR vs kNN on the same test set
  Problem Generator -> job types the MLR predicts worst
  Predict           -> Performance Element (+ Learning Element when finished jobs come back)
  Logs              -> logs/simple_predictions.csv and logs/simple_evaluations.csv
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from common import LOG_DIR, NUMERIC, CATEGORICAL
from make_batch import make_batch
from simple_mlr_agent import LearningAgent_MLR

BATCH_FILE = Path('transcode_batch.csv')    # same file the terminal agent watches

st.set_page_config(page_title='MLR vs kNN Agent', layout='wide')


@st.cache_resource(show_spinner='Training MLR and kNN on 55,027 jobs...')
def get_agent():
    # Trained once and shared across reruns, so feedback retraining persists until "Retrain from scratch"
    agent = LearningAgent_MLR()
    agent.load_data()
    agent.data_preprocessing()
    agent.train_agents()
    agent.results = agent.score_models()
    return agent


def run_batch(agent, df_new, source):
    """Predict a batch, learn from any measured utime, and keep the result for display."""
    try:
        X_new, job_warnings = agent.predict_jobs(df_new)
    except Exception as e:
        st.error(f'Could not process {source}: {e}')
        return
    learned = 0
    if 'utime' in df_new.columns:
        learned = int(pd.to_numeric(df_new['utime'], errors='coerce').notna().sum())
        X_new['utime_actual'] = pd.to_numeric(df_new['utime'], errors='coerce').values
        agent.learn_from_feedback(X_new[NUMERIC + CATEGORICAL], df_new['utime'])
    st.session_state.last_batch = {'source': source, 'X_new': X_new,
                                   'warnings': job_warnings, 'learned': learned}


agent = get_agent()
results = agent.results

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title('MLR Learning Agent')
    st.caption('Predicts how long a cloud video-transcoding job takes (utime, seconds) '
               'and compares Multilinear Regression to the tuned kNN (k=3, distance).')
    train_rows = st.empty()     # filled at the end, after any feedback retraining
    st.metric('Test rows', f'{len(agent.X_test):,}')
    if st.button('Retrain from scratch'):
        st.cache_resource.clear()
        st.session_state.pop('last_batch', None)
        st.rerun()
    st.divider()
    st.markdown('**Learning-agent components**\n'
                '- Performance Element: *Predict*\n'
                '- Learning Element: training + feedback\n'
                '- Critic: *Critic*\n'
                '- Problem Generator: *Problem Generator*')

tab_data, tab_critic, tab_problems, tab_predict, tab_logs = st.tabs(
    ['Data', 'Critic', 'Problem Generator', 'Predict', 'Logs'])

# ---------------- Data ----------------
with tab_data:
    st.header('Percepts: measured transcoding jobs')
    c1, c2, c3 = st.columns(3)
    c1.metric('Jobs', f'{len(agent.X):,}')
    c2.metric('Median utime', f'{agent.y.median():.1f} s')
    c3.metric('Mean utime', f'{agent.y.mean():.1f} s')
    st.caption('`id` and `umem` are dropped: memory use is only known after the job runs (leakage).')
    st.dataframe(agent.X.head(10))
    left, right = st.columns([3, 2])
    left.pyplot(agent.plot_codec_boxplot())
    right.subheader('utime summary')
    right.dataframe(agent.y.describe().round(2))
    right.caption('Output codec alone shifts run time a lot; h264 output is the slowest.')

# ---------------- Critic ----------------
with tab_critic:
    st.header('Critic: both models on the same test set')
    winner = 'kNN' if results['kNN']['RMSE'] < results['MLR']['RMSE'] else 'MLR'
    st.success(f'Better model on this test set: **{winner}** (lower RMSE)')

    cols = st.columns(4)
    for col, metric, better_low in zip(cols, ['MAE', 'RMSE', 'R2'], [True, True, False]):
        mlr_v, knn_v = results['MLR'][metric], results['kNN'][metric]
        col.metric(f'{metric}: kNN', f'{knn_v:.3f}', f'{knn_v - mlr_v:+.3f} vs MLR',
                   delta_color='inverse' if better_low else 'normal')
    off_mlr, off_knn = agent.share_far_off(agent.pred_mlr), agent.share_far_off(agent.pred_knn)
    cols[3].metric('>50% off: kNN', f'{off_knn:.1%}', f'{off_knn - off_mlr:+.1%} vs MLR', delta_color='inverse')

    st.dataframe(pd.DataFrame(results).round(3))
    st.pyplot(agent.plot_comparison(results))
    st.caption('Top: points on the red line are perfect predictions. '
               'Bottom: residuals should form a flat band around 0; above 0 means the model under-predicted.')

# ---------------- Problem Generator ----------------
with tab_problems:
    st.header('Problem Generator: where MLR struggles most')
    n = st.slider('Job types to show', 3, 15, 3)
    worst = agent.worst_job_types(n)
    st.bar_chart(worst, horizontal=True, stack=False, y_label='', x_label='Mean absolute error (s)')
    st.dataframe(worst.round(2))
    for job_type in worst.index[:3]:
        st.write(f'Suggestion: run more **{job_type}** jobs to learn this conversion better.')

# ---------------- Predict ----------------
with tab_predict:
    st.header('Performance Element: predict new jobs')

    if 'custom_jobs' not in st.session_state:
        sample = agent.X_test.sample(1).astype(object)
        st.session_state.custom_jobs = sample.assign(utime=None).astype(str).reset_index(drop=True)
    edited = st.data_editor(st.session_state.custom_jobs, num_rows='dynamic', key='job_editor')
    c1, c2 = st.columns(2)
    if c1.button('Predict these jobs', type='primary'):
        jobs = edited.replace({'None': None, '': None})
        if jobs['utime'].isna().all():
            jobs = jobs.drop(columns='utime')
        run_batch(agent, jobs, 'custom jobs')
    if c2.button('New random job'):
        del st.session_state.custom_jobs
        st.rerun()

    st.subheader('Simulated environment')
    c1, c2, c3 = st.columns(3)
    if c1.button('Simulate batch'):
        run_batch(agent, make_batch(), 'simulated batch')
    if c2.button('Simulate finished batch (feedback)'):
        run_batch(agent, make_batch(finished=True), 'simulated finished batch')
    if c3.button(f'Pick up {BATCH_FILE}', disabled=not BATCH_FILE.exists(),
                 help='Consumes the file written by `python make_batch.py`, then deletes it.'):
        run_batch(agent, pd.read_csv(BATCH_FILE), str(BATCH_FILE))
        BATCH_FILE.unlink()
    uploaded = st.file_uploader('...or upload a batch CSV', type='csv')
    if uploaded is not None and st.button('Predict uploaded batch'):
        run_batch(agent, pd.read_csv(uploaded), uploaded.name)

    batch = st.session_state.get('last_batch')
    if batch:
        st.subheader(f"Results: {batch['source']}")
        X_new = batch['X_new']
        shown = ['codec', 'o_codec', 'o_width', 'o_height', 'MLR_pred', 'kNN_pred']
        if 'utime_actual' in X_new:
            shown.append('utime_actual')
        table = X_new[shown].assign(warnings=[len(w) for w in batch['warnings']])
        st.dataframe(table.style.highlight_between(subset='warnings', left=1, color='#f5c26b55'))
        for i, warnings in enumerate(batch['warnings']):
            for w in warnings:
                st.warning(f'Job {i}: {w}')
        if batch['learned']:
            st.success(f"Learned from {batch['learned']} finished job(s); both models retrained "
                       f'on {len(agent.X_train):,} rows.')

# ---------------- Logs ----------------
with tab_logs:
    st.header('Logs')
    for name in ['simple_predictions.csv', 'simple_evaluations.csv']:
        path = LOG_DIR / name
        st.subheader(name)
        if path.exists():
            log = pd.read_csv(path)
            st.dataframe(log.tail(200).iloc[::-1])
            st.download_button(f'Download {name}', path.read_bytes(), file_name=name)
        else:
            st.info('Nothing logged yet.')

train_rows.metric('Training rows', f'{len(agent.X_train):,}',
                  help='Grows when finished jobs are fed back (Learning Element).')
