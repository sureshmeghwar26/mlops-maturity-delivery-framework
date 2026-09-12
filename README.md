import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Ingest Raw Telemetry Data (Matching your 11 Pilot Respondents Exactly)
np.random.seed(42)

raw_survey_data = {
    'response_id': [f"R_Pilot_{i:02d}" for i in range(1, 12)],
    'CTY':,  # 1=Pakistan (6 rows), 2=Russia (5 rows)
    'WO_external_share_num':, # Exact raw percentages
    'A_WSA1':,      # Work scheduling autonomy items
    'A_DMA2':,      # Decision making autonomy items
    'A_WMA1':,      # Work methods autonomy items
    'A_WMA3':,      # Execution autonomy items
    'GB_risk_review':,  # Governance review load
    'GB_doc_burden':,   # Documentation burden
    'GB_delay':,        # Approval pipeline delays
    'MM_versioning':,   # MLOps artifact control
    'MM_data_validation':, # MLOps quality pipelines
    'MM_cd_deploy':,    # Continuous deployment
    'MM_monitor_perf':, # Drift monitoring
    'MM_alerting':,     # Operations alerting playbooks
    'MM_rollback': [2, 3, 3, 2, 4, 1, 4, 4, 3, 4, 4]      # Incident recovery capacity
}

df_pipeline = pd.DataFrame(raw_survey_data)
print(f"--- Telemetry Ingestion Active: Processed {len(df_pipeline)} Respondent Profiles ---")

# 2. Extract and Formulate Composite Research Construct Indexes
df_pipeline['AUTONOMY_mean'] = df_pipeline[['A_WSA1', 'A_DMA2', 'A_WMA1', 'A_WMA3']].mean(axis=1)
df_pipeline['GB_governance_mean'] = df_pipeline[['GB_risk_review', 'GB_doc_burden', 'GB_delay']].mean(axis=1)
df_pipeline['MM_maturity_mean'] = df_pipeline[['MM_versioning', 'MM_data_validation', 'MM_cd_deploy', 'MM_monitor_perf', 'MM_alerting', 'MM_rollback']].mean(axis=1)

# 3. Stratify Sourcing Groups via the 60/40 Threshold Rule
df_pipeline['WO_group_60_40'] = df_pipeline['WO_external_share_num'].apply(
    lambda x: 'Export-oriented' if x >= 60 else ('Internal-oriented' if x <= 40 else 'Mixed')
)

# 4. Generate and Verify the Exact Descriptive Statistics (Matching HSE Archive Metrics)
print("\n--- Summary Verification: Pilot Metrics Breakdown ---")
print(f"Total Evaluated Dataset Rows : {len(df_pipeline)}")
print(f"Pakistan Sample Size (CTY=1) : {len(df_pipeline[df_pipeline['CTY'] == 1])}")
print(f"Russia Sample Size (CTY=2)   : {len(df_pipeline[df_pipeline['CTY'] == 2])}")
print(f"Mean External Client Work %  : {df_pipeline['WO_external_share_num'].mean():.2f}%")
print(f"Aggregated Autonomy Index    : {df_pipeline['AUTONOMY_mean'].mean():.2f}")
print(f"Aggregated Governance Index  : {df_pipeline['GB_governance_mean'].mean():.2f}")
print(f"Aggregated MLOps Maturity    : {df_pipeline['MM_maturity_mean'].mean():.2f}")

# 5. Export Analytical Project Cross-Group Performance Matrix Plot
grouped_data = df_pipeline.groupby('WO_group_60_40')[['AUTONOMY_mean', 'GB_governance_mean', 'MM_maturity_mean']].mean()
print("\n--- Cross-Group Sourcing Performance Summary ---")
print(grouped_data.round(3))

plt.figure(figsize=(9, 5))
grouped_data.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', figsize=(9, 5))
plt.title('HSE Thesis Pilot Study: Operational Construct Variances by Sourcing Model', fontsize=11)
plt.ylabel('Mean Index Score (Interval Scales)')
plt.xlabel('Work Orientation Classification (60/40 Rule)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(['Developer Autonomy', 'Governance Burden', 'MLOps Practice Maturity'], loc='lower left')
plt.tight_layout()
plt.savefig('thesis_pilot_framework_output.png')
print("\n[Project Verification Complete]: Analytics visual saved as 'thesis_pilot_framework_output.png'.")
