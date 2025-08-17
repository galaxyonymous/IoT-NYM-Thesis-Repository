# run_nym_unsup.py
import pandas as pd
import sys
import traceback
from nym_pipeline import (
    FEATURES, load_nym_dataset, standardize,
    run_hdbscan, run_gmm_bic, run_spectral_eigengap,
    bootstrap_ari, proxy_classification, confusion_for_model
)

def main():
    try:
        csv_path = "data/nym_metadata.csv"  # update path if needed
        print("Loading dataset...")
        df = load_nym_dataset(csv_path, FEATURES)
        X = standardize(df)  # standardize once for all unsupervised methods

        print(f"Loaded {len(df)} samples with {len(FEATURES)} features")
        print(f"Features: {FEATURES}")
        print("=" * 80)

        # --- HDBSCAN / DBSCAN fallback
        print("Running HDBSCAN/DBSCAN clustering...")
        hdb = run_hdbscan(X)
        print("Computing stability...")
        hdb_stability = bootstrap_ari(X, lambda Xsub: run_hdbscan(Xsub).labels)
        print(f"[HDBSCAN/DBSCAN] params: {hdb.params}")
        print(f"[HDBSCAN/DBSCAN] clusters: {hdb.n_clusters}, noise_rate: {hdb.noise_rate:.3f}")
        print(f"[HDBSCAN/DBSCAN] metrics: {hdb.metrics}")
        print(f"[HDBSCAN/DBSCAN] stability_ari: {hdb_stability:.3f}")

        # --- GMM (BIC)
        print("\nRunning GMM with BIC selection...")
        gmm = run_gmm_bic(X)
        print(f"[GMM-BIC] k: {gmm.n_clusters}, params: {gmm.params}")
        print(f"[GMM-BIC] metrics: {gmm.metrics}")

        # --- Spectral (eigengap via silhouette)
        print("\nRunning Spectral clustering...")
        spc = run_spectral_eigengap(X)
        print(f"[Spectral] k: {spc.n_clusters}, params: {spc.params}")
        print(f"[Spectral] metrics: {spc.metrics}")

        # --- Summary table
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        
        summary_data = []
        for result in [hdb, gmm, spc]:
            row = {
                "Method": result.name,
                "Clusters": result.n_clusters,
                "Noise %": f"{result.noise_rate*100:.1f}%" if result.noise_rate is not None else "N/A",
                "Silhouette": f"{result.metrics.get('silhouette', float('nan')):.3f}",
                "Davies-Bouldin": f"{result.metrics.get('davies_bouldin', float('nan')):.3f}",
                "Calinski-Harabasz": f"{result.metrics.get('calinski_harabasz', float('nan')):.1f}",
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Add stability for HDBSCAN
        print(f"\nHDBSCAN Stability (Bootstrap ARI): {hdb_stability:.3f}")

        # --- Proxy separability on the best label set (choose by silhouette)
        print("\n" + "=" * 80)
        print("PROXY SEPARABILITY ANALYSIS")
        print("=" * 80)
        
        cand = [
            ("HDBSCAN/DBSCAN", hdb.labels, hdb.metrics.get("silhouette", float("nan"))),
            ("GMM-BIC",        gmm.labels, gmm.metrics.get("silhouette", float("nan"))),
            ("Spectral",       spc.labels, spc.metrics.get("silhouette", float("nan"))),
        ]
        cand = sorted(cand, key=lambda t: (t[2] if t[2]==t[2] else -1), reverse=True)  # NaN-safe
        best_name, best_labels, best_sil = cand[0]
        print(f"Using labels from {best_name} (silhouette={best_sil:.3f})")

        print("Running proxy classification...")
        proxy_df = proxy_classification(X, best_labels)
        print("\nProxy separability (weighted %):")
        print(proxy_df.to_string(index=False))

        # Example: confusion matrix for SVM pseudo-label model
        print(f"\nComputing confusion matrix for {best_name} labels (SVM classifier)...")
        cm, classes = confusion_for_model(X, best_labels, model_name="SVM (RBF)")
        cm_df = pd.DataFrame(cm, 
                            index=[f"true_{c}" for c in classes], 
                            columns=[f"pred_{c}" for c in classes])
        print(cm_df.to_string())

        # --- Save results to file
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Save summary table
        summary_df.to_csv("clustering_summary.csv", index=False)
        print("Saved clustering_summary.csv")
        
        # Save proxy classification results
        proxy_df.to_csv("proxy_classification_results.csv", index=False)
        print("Saved proxy_classification_results.csv")
        
        # Save confusion matrix
        cm_df.to_csv("confusion_matrix.csv")
        print("Saved confusion_matrix.csv")
        
        print("\nAnalysis complete! Check the CSV files for detailed results.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
