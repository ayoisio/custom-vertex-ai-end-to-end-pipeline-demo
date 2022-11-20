# Vertex AI Pipelines - Custom Model Deployment

![Successful pipeline execution graph](/img/successful_pipeline_graph.png)

## Dataset
The dataset (available for download [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly __imbalanced__, the positive class (frauds) account for 0.172% of all transactions.

## Workflow
1. Upload .csv or .parquet file of dataset to GCS
2. [Create BigLake table](https://cloud.google.com/bigquery/docs/query-cloud-storage-using-biglake) that reads from file 
2. [Build](https://cloud.google.com/sdk/gcloud/reference/builds/submit) model in custom container and [submit](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling) to Artifact Registry repo.
3. Build Vertex AI Pipeline
    1. Create dataset ([TabularDatasetCreateOp](https://google-cloud-pipeline-components.readthedocs.io/page/google_cloud_pipeline_components.v1.dataset.html#google_cloud_pipeline_components.v1.dataset.TabularDatasetCreateOp))
    2. Create custom container training job ([CustomContainerTrainingJob](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob))
    3. Remove target field from batch prediction input ([TargetFieldDataRemovalOp](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.27/google_cloud_pipeline_components.experimental.evaluation.html#google_cloud_pipeline_components.experimental.evaluation.TargetFieldDataRemoverOp))
    4. Run batch prediction ([ModelBatchPredictionOp](https://google-cloud-pipeline-components.readthedocs.io/page/google_cloud_pipeline_components.v1.batch_predict_job.html#google_cloud_pipeline_components.v1.batch_predict_job.ModelBatchPredictOp))
    5. Perform classification evaluation ([ModelEvaluationClassificationOp](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.27/google_cloud_pipeline_components.experimental.evaluation.html#google_cloud_pipeline_components.experimental.evaluation.ModelEvaluationClassificationOp)) 
    6. Perform feature attribution ([ModelEvaluationFeatureAttributionOp](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.27/google_cloud_pipeline_components.experimental.evaluation.html#google_cloud_pipeline_components.experimental.evaluation.ModelEvaluationFeatureAttributionOp))
    7. Import model evaluation task ([ModelImportEvaluationOp](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.27/google_cloud_pipeline_components.experimental.evaluation.html#google_cloud_pipeline_components.experimental.evaluation.ModelImportEvaluationOp))
    8. Make deployment decision (auPrc > 95%)
    9. Create model endpoint ([EndpointCreateOp](https://google-cloud-pipeline-components.readthedocs.io/page/google_cloud_pipeline_components.v1.endpoint.html#google_cloud_pipeline_components.v1.endpoint.EndpointCreateOp))
    10. Deploy model ([ModelDeployOp](https://google-cloud-pipeline-components.readthedocs.io/page/google_cloud_pipeline_components.v1.endpoint.html#google_cloud_pipeline_components.v1.endpoint.ModelDeployOp))
    11. Send email notification ([VertexNotificationEmailOp](https://cloud.google.com/vertex-ai/docs/pipelines/email-notification-component))
  
  
