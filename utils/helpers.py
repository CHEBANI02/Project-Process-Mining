import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py

def load_data():
    # Load all necessary data
    df = pd.read_csv('data/event_log_df.csv')
    preprocessed_df = pd.read_csv('data/preprocessed_df.csv')
    X_train = np.load('data/X_train.npy', allow_pickle=True)
    y_train = np.load('data/y_train.npy', allow_pickle=True)
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    max_len = np.load('data/max_len.npy', allow_pickle=True).item()
    activity_encoder_classes = np.load('data/activity_encoder_classes.npy', allow_pickle=True)
    full_sequences = np.load('data/full_sequences.npy', allow_pickle=True)
    
    return {
        'df': df,
        'preprocessed_df': preprocessed_df,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'max_len': max_len,
        'activity_encoder_classes': activity_encoder_classes,
        'full_sequences': full_sequences
    }

def load_model_data():
    model = load_model('models/bilstm_final_EMLogmodel.h5')
    activity_names = [
        'Admission', 'Reservation application', 'Imaging planning', 'Diagnosis',
        'Consultation application', 'Consultation', 'Prescription given', 'Consultation arrangement',
        'Consultation summary', 'Register', 'Pre-examination', 'Perform Triage',
        'Medicine accounting', 'Medicine taking', 'Medicine packing', 'Payment notice of medicine',
        'Payment2', 'Payment notice of X-ray', 'Payment1', 'Reservation',
        'Imaging register', 'Start imaging', 'Machine operation', 'Image processing',
        'Backup', 'Write report', 'Receiving patients', 'Perform Rescue',
        'Pay the fees', 'Payment notice of treatment', 'Payment3', 'File storage'
    ]
    
    task_mapping = {i+1: name for i, name in enumerate(activity_names)}
    reverse_task_mapping = {v.lower(): k for k, v in task_mapping.items()}
    
    return {
        'model': model,
        'task_mapping': task_mapping,
        'reverse_task_mapping': reverse_task_mapping,
        'activity_names': activity_names
    }

def get_case_details(df, case_id):
    return df[df['CaseID'] == case_id].sort_values('Timestamp')

def generate_process_map(df, clustered=False, start_date=None, end_date=None):
    # Make sure Timestamp is datetime type
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter by date if provided
    if start_date and end_date:
        # Convert date inputs to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Cluster activities if requested
    if clustered:
        task_cluster_mapping = {
            # Your clustering mapping here (same as in your original code)
        }
        df = df.copy()
        df['Activity'] = df['Activity'].map(task_cluster_mapping)
    
    # Convert to event log and generate process map
    event_log = pm4py.format_dataframe(df, 
                                      case_id='CaseID',
                                      activity_key='Activity',
                                      timestamp_key='Timestamp')
    
    if clustered:
        heu_net = pm4py.discover_heuristics_net(event_log)
        pm4py.save_vis_heuristics_net(heu_net, 'assets/process_map_clustered.png')
        return 'assets/process_map_clustered.png'
    else:
        heu_net = pm4py.discover_heuristics_net(event_log)
        pm4py.save_vis_heuristics_net(heu_net, 'assets/process_map.png')
        return 'assets/process_map.png'

def predict_next_activity(model, input_sequence, max_len, reverse_task_mapping, task_mapping):
    try:
        input_activities = [x.strip().lower() for x in input_sequence.split(',')]
        input_codes = [reverse_task_mapping[act] for act in input_activities]
        input_seq_padded = pad_sequences([input_codes], maxlen=max_len, padding='post', value=0)
        prediction = model.predict(input_seq_padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0] + 1
        confidence = np.max(prediction)
        return task_mapping.get(predicted_class), confidence
    except Exception as e:
        return str(e), 0.0