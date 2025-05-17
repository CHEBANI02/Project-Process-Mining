import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #e6e9ef; 
        border-radius: 4px 4px 0 0; 
        gap: 1px; 
        padding-top: 10px; 
        padding-bottom: 10px; 
        font-weight: bold; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #007bff; 
        color: white; 
    }
    .stButton>button { 
        background-color: #007bff; 
        color: white; 
        border-radius: 5px; 
    }
    .stButton>button:hover { 
        background-color: #0056b3; 
    }
    .stMetric { 
        background-color: #e6f3ff; 
        padding: 10px; 
        border-radius: 5px; 
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to check file existence
def check_file(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return False
    return True

# Load model and data
@st.cache_resource
def load_model_and_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    assets_dir = os.path.join(base_dir, 'assets')
    
    # Define file paths
    model_path = os.path.join(models_dir, 'bilstm_final_EMLogmodel.h5')
    event_log_path = os.path.join(data_dir, 'event_log_df.csv')
    preprocessed_df_path = os.path.join(data_dir, 'preprocessed_df.csv')
    validation_results_path = os.path.join(data_dir, 'validation_results.csv')
    activity_encoder_classes_path = os.path.join(data_dir, 'activity_encoder_classes.npy')
    max_len_path = os.path.join(data_dir, 'max_len.npy')
    X_test_path = os.path.join(data_dir, 'X_test.npy')
    y_test_path = os.path.join(data_dir, 'y_test.npy')
    history_path = os.path.join(data_dir, 'training_history.npy')
    process_map_path = os.path.join(assets_dir, 'process_map.png')
    process_map_clustered_path = os.path.join(assets_dir, 'process_map_clustered.png')
    
    # Check if all files exist
    required_files = [model_path, event_log_path, preprocessed_df_path, validation_results_path,
                      activity_encoder_classes_path, max_len_path, X_test_path, y_test_path,
                      history_path, process_map_path, process_map_clustered_path]
    for path in required_files:
        if not check_file(path):
            return None, None, None, None, None, None, None, None, None, None
    
    try:
        model = load_model(model_path)
        # Parse timestamps in ISO 8601 format
        event_log_df = pd.read_csv(event_log_path, parse_dates=['Timestamp'])
        preprocessed_df = pd.read_csv(preprocessed_df_path, parse_dates=['Timestamp'])
        activity_encoder_classes = np.load(activity_encoder_classes_path, allow_pickle=True)
        max_len = np.load(max_len_path, allow_pickle=True).item()
        X_test = np.load(X_test_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)
        history = np.load(history_path, allow_pickle=True).item()
        validation_results = pd.read_csv(validation_results_path)
        
        # Ensure timestamps are in datetime format and handle timezone
        event_log_df['Timestamp'] = pd.to_datetime(event_log_df['Timestamp'], utc=True).dt.tz_convert(None)
        preprocessed_df['Timestamp'] = pd.to_datetime(preprocessed_df['Timestamp'], utc=True).dt.tz_convert(None)
        
        return model, event_log_df, preprocessed_df, activity_encoder_classes, max_len, X_test, y_test, history, validation_results, [process_map_path, process_map_clustered_path]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None

model, event_log_df, preprocessed_df, activity_encoder_classes, max_len, X_test, y_test, history, validation_results, process_map_paths = load_model_and_data()

# Check if loading was successful
if model is None:
    st.stop()

# Task mapping
task_mapping = {
    't1': 'Admission', 't2': 'Reservation application', 't3': 'Imaging planning', 't4': 'Diagnosis',
    't5': 'Consultation application', 't6': 'Consultation', 't7': 'Prescription given', 't8': 'Consultation arrangement',
    't9': 'Consultation summary', 't10': 'Register', 't11': 'Pre-examination', 't12': 'Perform Triage',
    't13': 'Medicine accounting', 't14': 'Medicine taking', 't15': 'Medicine packing', 't16': 'Payment notice of medicine',
    't17': 'Payment2', 't18': 'Payment notice of X-ray', 't19': 'Payment1', 't20': 'Reservation',
    't21': 'Imaging register', 't22': 'Start imaging', 't23': 'Machine operation', 't24': 'Image processing',
    't25': 'Backup', 't26': 'Write report', 't27': 'Receiving patients', 't28': 'Perform Rescue',
    't29': 'Pay the fees', 't30': 'Payment notice of treatment', 't31': 'Payment3', 't32': 'File storage'
}
activity_to_code = {v.lower(): int(k.replace('t', '')) for k, v in task_mapping.items()}
code_to_activity = {int(k.replace('t', '')): v for k, v in task_mapping.items()}

# Streamlit app layout
st.title("Healthcare Process Mining Framework")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Case Details", "Process Map", "Predict Next Activity", "Model Evaluation"])

with tab1:
    st.header("Case Details")
    st.markdown("Explore and filter case details from the healthcare event log. View case sequences and summary statistics.")
    
    # Initialize session state for visualization
    if 'case_visualized' not in st.session_state:
        st.session_state.case_visualized = False
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        case_id_filter = st.multiselect("Filter by Case ID", options=preprocessed_df['CaseID'].unique())
    with col2:
        activity_filter = st.multiselect("Filter by Activity", options=preprocessed_df['Activity'].unique())
    with col3:
        department_filter = st.multiselect("Filter by Department", options=preprocessed_df['Department'].dropna().unique())
    
    # Apply filters
    filtered_df = preprocessed_df.copy()
    if case_id_filter:
        filtered_df = filtered_df[filtered_df['CaseID'].isin(case_id_filter)]
    if activity_filter:
        filtered_df = filtered_df[filtered_df['Activity'].isin(activity_filter)]
    if department_filter:
        filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]
    
    # Summary statistics
    st.subheader("Case Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", filtered_df['CaseID'].nunique())
    col2.metric("Total Events", len(filtered_df))
    col3.metric("Unique Activities", filtered_df['Activity'].nunique())
    
    # Display DataFrame
    st.subheader("Event Log")
    st.dataframe(filtered_df, height=300)
    
    # Case sequence visualization button
    if st.button("Visualize Case Sequence", key="visualize_button"):
        st.session_state.case_visualized = True
    else:
        st.session_state.case_visualized = False
    
    # Case sequence visualization
    if st.session_state.case_visualized and case_id_filter and 'Timestamp' in filtered_df.columns:
        st.subheader("Case Sequence Visualization")
        selected_case = case_id_filter[0]
        case_df = filtered_df[filtered_df['CaseID'] == selected_case].sort_values('Timestamp')
        if not case_df.empty and case_df['Timestamp'].notna().any():
            # Drop rows with invalid timestamps
            case_df = case_df.dropna(subset=['Timestamp'])
            
            # Create an end time for visualization by using the next event's start time
            case_df = case_df.reset_index(drop=True)
            case_df['End_Timestamp'] = case_df['Timestamp'].shift(-1)
            # For the last event, assume a duration (e.g., 1 hour)
            last_end_time = case_df['Timestamp'].iloc[-1] + pd.Timedelta(hours=1)
            case_df.loc[case_df.index[-1], 'End_Timestamp'] = last_end_time
            
            # Calculate duration for each event
            case_df['Duration'] = (case_df['End_Timestamp'] - case_df['Timestamp']).dt.total_seconds() / 3600  # Duration in hours
            
            # Create the timeline with enhanced visualization
            fig = px.timeline(case_df,
                             x_start='Timestamp',
                             x_end='End_Timestamp',
                             y='Activity',
                             color='Activity',
                             title=f"Activity Sequence for Case {selected_case}",
                             hover_data=['Department', 'Resource', 'Duration'],
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_yaxes(autorange="reversed")
            fig.update_traces(marker_line_width=2, opacity=0.9)
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Activity",
                xaxis=dict(
                    tickformat="%Y-%m-%d %H:%M:%S",
                    autorange=True,
                    gridcolor='lightgrey'
                ),
                yaxis=dict(gridcolor='lightgrey'),
                legend_title_text='Activity',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the time range for clarity
            time_range = f"Time Range: {case_df['Timestamp'].min()} to {case_df['Timestamp'].max()}"
            st.write(time_range)
            
            # Validate time display
            if (case_df['End_Timestamp'] > case_df['Timestamp']).all():
                st.write("Verification: Timestamps and durations are correctly displayed for each event.")
            else:
                st.warning("Error: Some events have invalid durations. Check timestamp data.")
        else:
            st.warning("No valid timestamp data available for the selected case.")
    elif st.session_state.case_visualized and not case_id_filter:
        st.warning("Please select at least one Case ID to visualize.")

    # Download filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="filtered_case_details.csv",
        mime="text/csv"
    )

with tab2:
    st.header("Process Map")
    st.markdown("Visualize the process flow using heuristics nets for both original and clustered activities.")
    
    # Tabs for original and clustered maps
    subtab1, subtab2 = st.tabs(["Original Process Map", "Clustered Process Map"])
    
    with subtab1:
        if check_file(process_map_paths[0]):
            image = Image.open(process_map_paths[0])
            st.image(image, caption="Original Process Map", use_container_width=True)
        else:
            st.error("Original process map image not found.")
    
    with subtab2:
        if check_file(process_map_paths[1]):
            image = Image.open(process_map_paths[1])
            st.image(image, caption="Clustered Process Map", use_container_width=True)
        else:
            st.error("Clustered process map image not found.")

with tab3:
    st.header("Predict Next Activity")
    st.markdown("Input a sequence of activities to predict the next activity using the Bi-LSTM model.")
    
    # Initialize session state for selected activities and prediction
    if 'selected_activities' not in st.session_state:
        st.session_state.selected_activities = ["Admission", "Imaging planning"]
    if 'predicted_result' not in st.session_state:
        st.session_state.predicted_result = None
    
    # Activity selection
    st.write("Select activities in sequence (use the order of occurrence):")
    selected_activities = st.multiselect(
        "Activities",
        options=list(task_mapping.values()),
        default=st.session_state.selected_activities,
        key="activity_multiselect"
    )
    
    # Update session state for selected activities
    st.session_state.selected_activities = selected_activities
    
    # Clear previous prediction when activities change
    if 'last_selected_activities' not in st.session_state:
        st.session_state.last_selected_activities = selected_activities
    if st.session_state.last_selected_activities != selected_activities:
        st.session_state.predicted_result = None
        st.session_state.last_selected_activities = selected_activities
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Predict", key="predict_button"):
            if selected_activities:
                try:
                    # Convert to codes
                    input_codes = [activity_to_code[act.lower()] for act in selected_activities]
                    input_seq_padded = pad_sequences([input_codes], maxlen=max_len, padding='post', value=0)
                    
                    # Predict
                    with st.spinner("Predicting..."):
                        prediction = model.predict(input_seq_padded, verbose=0)[0]
                    predicted_class = np.argmax(prediction) + 1
                    predicted_activity = code_to_activity.get(predicted_class, "Unknown")
                    
                    # Prediction probabilities
                    top_k = 5
                    top_indices = np.argsort(prediction)[::-1][:top_k]
                    probabilities = prediction[top_indices]
                    top_activities = [code_to_activity.get(i + 1, "Unknown") for i in top_indices]
                    
                    # Store result in session state
                    st.session_state.predicted_result = {
                        'activity': predicted_activity,
                        'top_activities': top_activities,
                        'probabilities': probabilities
                    }
                except KeyError as e:
                    st.error(f"Activity '{e.args[0]}' not found in mapping")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.warning("Please select at least one activity.")
    
    # Display prediction only after button press
    if st.session_state.predicted_result:
        st.success(f"Predicted Next Activity: **{st.session_state.predicted_result['activity']}**")
        prob_df = pd.DataFrame({
            'Activity': st.session_state.predicted_result['top_activities'],
            'Probability': st.session_state.predicted_result['probabilities']
        })
        fig = px.bar(prob_df, x='Probability', y='Activity', orientation='h',
                    title='Top 5 Predicted Activities with Probabilities',
                    color='Probability',
                    color_continuous_scale='Viridis',
                    text='Probability')
        fig.update_traces(texttemplate='%{text:.2%}', textposition='auto')
        fig.update_layout(
            yaxis={'autorange': 'reversed'},
            xaxis_title='Probability',
            yaxis_title='Activity',
            coloraxis_colorbar_title='Probability',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("Reset", key="reset_button"):
            st.session_state.selected_activities = ["Admission", "Imaging planning"]
            st.session_state.last_selected_activities = ["Admission", "Imaging planning"]
            if 'predicted_result' in st.session_state:
                del st.session_state.predicted_result
            st.rerun()

with tab4:
    st.header("Model Evaluation")
    st.markdown("Evaluate the Bi-LSTM model's performance on the test set with comprehensive metrics and visualizations.")
    
    # Initialize session state for evaluation
    if 'evaluated' not in st.session_state:
        st.session_state.evaluated = False
    
    if st.button("Evaluate Model", key="evaluate_button"):
        st.session_state.evaluated = True
    else:
        st.session_state.evaluated = False
    
    if st.session_state.evaluated:
        # Training history
        st.subheader("Training History")
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
        
        # Accuracy plot
        fig.add_trace(go.Scatter(x=list(range(1, len(history['accuracy']) + 1)),
                               y=history['accuracy'],
                               mode='lines+markers',
                               name='Training Accuracy',
                               line=dict(color='#1f77b4')),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, len(history['val_accuracy']) + 1)),
                               y=history['val_accuracy'],
                               mode='lines+markers',
                               name='Validation Accuracy',
                               line=dict(color='#ff7f0e')),
                     row=1, col=1)
        max_acc = max(max(history['accuracy']), max(history['val_accuracy']))
        fig.add_annotation(text=f"Max Acc: {max_acc:.4f}",
                         x=len(history['accuracy']), y=max_acc,
                         showarrow=True, arrowhead=1, row=1, col=1)
        
        # Loss plot
        fig.add_trace(go.Scatter(x=list(range(1, len(history['loss']) + 1)),
                               y=history['loss'],
                               mode='lines+markers',
                               name='Training Loss',
                               line=dict(color='#1f77b4')),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(1, len(history['val_loss']) + 1)),
                               y=history['val_loss'],
                               mode='lines+markers',
                               name='Validation Loss',
                               line=dict(color='#ff7f0e')),
                     row=1, col=2)
        min_loss = min(min(history['loss']), min(history['val_loss']))
        fig.add_annotation(text=f"Min Loss: {min_loss:.4f}",
                         x=len(history['loss']), y=min_loss,
                         showarrow=True, arrowhead=1, row=1, col=2)
        
        fig.update_layout(height=400, width=800, title_text="Training History",
                         template='plotly_white', showlegend=True)
        st.plotly_chart(fig)
        
        # Model performance metrics
        st.subheader("Model Performance Metrics")
        y_test_adj = np.array(y_test, dtype=np.int32) - 1
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        
        test_loss, test_accuracy = model.evaluate(X_test, y_test_adj, verbose=0)
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{test_accuracy:.4f}")
        col2.metric("Test Loss", f"{test_loss:.4f}")
        
        # Classification report
        unique_labels = np.unique(np.concatenate((y_test_adj, y_pred)))
        report = classification_report(y_test_adj, y_pred, labels=unique_labels,
                                     target_names=[activity_encoder_classes[i] for i in unique_labels],
                                     output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        # Separate accuracy, macro avg, and weighted avg
        accuracy_row = pd.DataFrame([report['accuracy']], columns=['accuracy'], index=['accuracy'])
        macro_avg_row = pd.DataFrame([report['macro avg']], index=['macro avg'])
        weighted_avg_row = pd.DataFrame([report['weighted avg']], index=['weighted avg'])
        final_report_df = pd.concat([report_df.drop(['accuracy', 'macro avg', 'weighted avg']),
                                   accuracy_row, macro_avg_row, weighted_avg_row])
        st.subheader("Classification Report")
        st.dataframe(final_report_df.style.format("{:.2f}"), height=400)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_adj, y_pred, labels=unique_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        fig = px.imshow(cm_normalized,
                       labels=dict(x="Predicted", y="Actual", color="Normalized Count"),
                       x=[activity_encoder_classes[i] for i in unique_labels],
                       y=[activity_encoder_classes[i] for i in unique_labels],
                       color_continuous_scale='Blues',
                       title='Normalized Confusion Matrix',
                       text_auto='.2f',
                       aspect='auto')
        fig.update_layout(width=1000, height=800,
                         coloraxis_colorbar_title="Normalized Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Training summary
        st.subheader("Training Summary")
        last_epoch = len(history['accuracy'])
        last_train_acc = history['accuracy'][-1]
        last_val_acc = history['val_accuracy'][-1]
        last_train_loss = history['loss'][-1]
        last_val_loss = history['val_loss'][-1]
        st.write(f"Last Epoch: {last_epoch}")
        st.write(f"Training Accuracy: {last_train_acc:.4f} (Loss: {last_train_loss:.4f})")
        st.write(f"Validation Accuracy: {last_val_acc:.4f} (Loss: {last_val_loss:.4f})")
        
        # Class-wise accuracy
        results_df = pd.DataFrame({
            'Ground_Truth_Code': y_test_adj,
            'Predicted_Code': y_pred,
            'Ground_Truth': [activity_encoder_classes[i] for i in y_test_adj],
            'Predicted': [activity_encoder_classes[i] for i in y_pred],
            'Correct': y_test_adj == y_pred
        })
        class_acc = results_df.groupby('Ground_Truth')['Correct'].mean()
        st.subheader("Class-wise Accuracy")
        fig = px.bar(class_acc, x=class_acc.values, y=class_acc.index, orientation='h',
                    title='Class-wise Accuracy',
                    text=class_acc.values,
                    color=class_acc.values,
                    color_continuous_scale='Greens')
        fig.update_traces(texttemplate='%{text:.2%}', textposition='auto')
        fig.update_layout(xaxis_title="Accuracy", yaxis_title="Activity",
                         coloraxis_colorbar_title="Accuracy",
                         template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Metrics
        st.subheader("Additional Performance Metrics")
        # Micro-average confusion matrix for overall metrics
        cm_micro = confusion_matrix(y_test_adj, y_pred)
        TP = np.diag(cm_micro)
        FP = cm_micro.sum(axis=0) - TP
        FN = cm_micro.sum(axis=1) - TP
        TN = cm_micro.sum() - (TP + FP + FN)
        
        # Per-class metrics
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        sensitivity = recall  # Sensitivity is the same as recall
        specificity = TN / (TN + FP)
        fdr = FP / (TP + FP)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        
        # Handle division by zero
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
        f1_score = np.nan_to_num(f1_score, nan=0.0)
        sensitivity = np.nan_to_num(sensitivity, nan=0.0)
        specificity = np.nan_to_num(specificity, nan=0.0)
        fdr = np.nan_to_num(fdr, nan=0.0)
        mcc = np.nan_to_num(mcc, nan=0.0)
        
        # Micro-average metrics
        micro_precision = np.mean(precision)
        micro_recall = np.mean(recall)
        micro_f1 = np.mean(f1_score)
        micro_sensitivity = np.mean(sensitivity)
        micro_specificity = np.mean(specificity)
        micro_fdr = np.mean(fdr)
        micro_mcc = np.mean(mcc)
        
        # ROC AUC
        y_score = model.predict(X_test, verbose=0)
        y_test_bin = label_binarize(y_test_adj, classes=np.arange(len(activity_encoder_classes)))
        if y_test_bin.shape[0] * y_test_bin.shape[1] == y_score.shape[0] * y_score.shape[1]:
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = np.nan
            st.warning("Unable to compute ROC AUC due to shape mismatch.")
        
        # Display metrics table
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Sensitivity', 'Specificity', 'FDR', 'MCC'],
            'Value': [micro_precision, micro_recall, micro_f1, roc_auc, micro_sensitivity, micro_specificity, micro_fdr, micro_mcc]
        })
        st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}))

        # ROC Curve
        st.subheader("ROC Curve (Micro-average)")
        if not np.isnan(roc_auc):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                   name=f'Micro-average ROC (AUC = {roc_auc:.4f})',
                                   fill='tozeroy'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   line=dict(dash='dash'), showlegend=False))
            fig.update_layout(title='Micro-average ROC Curve',
                             xaxis_title='False Positive Rate',
                             yaxis_title='True Positive Rate',
                             template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

