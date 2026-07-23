"""Create WESAD plots, summary tables, and a LaTeX research-report project.

This reporting-only stage consumes the completed signalwise experiment output
and one representative raw participant record.  It does not alter training or
preprocessing data.  Run from the repository root:
``.venv\\Scripts\\python.exe 08_visualization_reporting\\generate_visualizations.py``.
"""
from __future__ import annotations

import json, pickle, shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
OUT = ROOT / "07_classical_ml" / "outputs" / "signalwise"
REPORT = ROOT / "report"
WESAD = ROOT / "Data" / "Raw" / "WESAD"
SUBJECT = "S2"
TARGET_FS = 4
SIGNALS = {"E4_ACC":("wrist","ACC",32),"E4_BVP":("wrist","BVP",64),"E4_EDA":("wrist","EDA",4),"E4_HR":("csv","HR",1),"E4_IBI":("csv","IBI",None),"E4_TEMP":("wrist","TEMP",4),"RB_ACC":("chest","ACC",700),"RB_ECG":("chest","ECG",700),"RB_EDA":("chest","EDA",700),"RB_EMG":("chest","EMG",700),"RB_RESP":("chest","Resp",700),"RB_TEMP":("chest","Temp",700)}
VARIANTS = ["raw", "filtered", "filtered_normalized"]

def save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

def aligned(values, fs, length):
    if fs and fs > TARGET_FS and fs % TARGET_FS == 0: return signal.resample_poly(values, 1, int(fs/TARGET_FS), axis=0)[:length]
    return signal.resample(values, length, axis=0)

def e4_csv(subject, name, length):
    folder=WESAD/subject/f"{subject}_E4_Data"
    if name=="HR": return signal.resample(pd.read_csv(folder/"HR.csv",header=None).iloc[2:,0].astype(float),length)
    d=pd.read_csv(folder/"IBI.csv",header=None,names=["t","v"]); d=d.apply(pd.to_numeric,errors="coerce").dropna()
    return np.interp(np.linspace(d.t.iloc[0],d.t.iloc[-1],length),d.t,d.v)

def raw_signals():
    with (WESAD/SUBJECT/f"{SUBJECT}.pkl").open("rb") as f: record=pickle.load(f,encoding="latin1")
    length=len(record["label"])//700*TARGET_FS
    result={}
    for name,(device,key,fs) in SIGNALS.items():
        v=e4_csv(SUBJECT,key,length) if device=="csv" else aligned(record["signal"][device][key],fs,length)
        result[name]=np.asarray(v)[:,0] if np.asarray(v).ndim==2 else np.asarray(v)
    labels=np.rint(signal.resample(record["label"],length)).astype(int)
    return result,labels

def raw_plots(data):
    summaries=[]
    for name, values in data.items():
        clean=values[np.isfinite(values)]; display=clean[::max(1,len(clean)//6000)]; zoom=clean[:min(len(clean),400)]
        f,ax=plt.subplots(figsize=(11,3.5)); ax.plot(np.arange(len(display))/TARGET_FS,display,lw=.7,label="raw"); ax.set(title=f"{name} entire aligned raw signal ({SUBJECT})",xlabel="Time (s)",ylabel="Amplitude");ax.grid(alpha=.3);ax.legend();save(f,PLOTS/"raw"/name,"entire_signal")
        f,ax=plt.subplots(figsize=(11,3.5));ax.plot(np.arange(len(zoom))/TARGET_FS,zoom,lw=1,label="first 100 s");ax.set(title=f"{name} zoomed raw segment",xlabel="Time (s)",ylabel="Amplitude");ax.grid(alpha=.3);ax.legend();save(f,PLOTS/"raw"/name,"zoom_segment")
        sample=np.random.default_rng(42).choice(clean,size=min(15000,len(clean)),replace=False)
        f,axes=plt.subplots(2,3,figsize=(13,7));sns.histplot(sample,kde=True,ax=axes[0,0]);axes[0,0].set_title("Histogram and KDE");sns.boxplot(y=sample,ax=axes[0,1]);axes[0,1].set_title("Box plot");sns.violinplot(y=sample,ax=axes[0,2]);axes[0,2].set_title("Violin plot"); x=np.sort(sample);axes[1,0].plot(x,np.arange(1,len(x)+1)/len(x));axes[1,0].set_title("Cumulative distribution");stats.probplot(sample,plot=axes[1,1]);axes[1,1].set_title("QQ plot");z=np.abs(stats.zscore(sample,nan_policy="omit"));axes[1,2].scatter(np.arange(len(sample)),sample,c=np.where(z>3,"tab:red","tab:blue"),s=3);axes[1,2].set_title("Outliers (|z| > 3)")
        for ax in axes.ravel():ax.grid(alpha=.25)
        f.suptitle(f"{name}: distribution and quality checks");f.tight_layout();save(f,PLOTS/"raw"/name,"distribution_quality")
        summaries.append({"signal":name,"samples":len(values),"missing":int(np.isnan(values).sum()),"mean":np.nanmean(values),"std":np.nanstd(values),"min":np.nanmin(values),"max":np.nanmax(values),"outlier_fraction":float(np.mean(np.abs(stats.zscore(sample))>3))})
    pd.DataFrame(summaries).to_csv(PLOTS/"raw"/"signal_statistics.csv",index=False)

def feature_and_ml_plots():
    comparison=pd.read_csv(OUT/"signalwise_model_comparison.csv")
    # Overall/signal-wise metric comparison.
    metrics=["accuracy","precision","recall","f1","roc_auc"]
    long=comparison.melt(id_vars=["experiment","variant"],value_vars=metrics,var_name="metric",value_name="value")
    f,ax=plt.subplots(figsize=(13,5));sns.barplot(data=long[long.experiment=="ALL_SIGNALS"],x="metric",y="value",hue="variant",ax=ax);ax.set(title="All-signal preprocessing comparison",ylabel="Score",xlabel="Metric",ylim=(0,1));ax.grid(axis="y",alpha=.3);save(f,PLOTS/"ml_results","overall_metric_comparison")
    f,ax=plt.subplots(figsize=(13,6));sns.barplot(data=comparison[comparison.experiment!="ALL_SIGNALS"],x="experiment",y="f1",hue="variant",ax=ax);ax.set(title="Signal-wise F1 comparison",xlabel="Signal",ylabel="Held-out F1");ax.tick_params(axis="x",rotation=45);ax.grid(axis="y",alpha=.3);save(f,PLOTS/"signal_comparison","signalwise_f1")
    for exp in comparison.experiment.unique():
        fig,axes=plt.subplots(1,3,figsize=(13,3.7));
        for ax,variant in zip(axes,VARIANTS):
            pred_path=OUT/exp/variant/"held_out_predictions.csv"
            if not pred_path.exists(): ax.set_axis_off();continue
            d=pd.read_csv(pred_path);cm=confusion_matrix(d.stress_label,d.predicted_stress,labels=[0,1]);sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax,xticklabels=["Non-stress","Stress"],yticklabels=["Non-stress","Stress"]);ax.set_title(variant);ax.set(xlabel="Predicted",ylabel="True")
        fig.suptitle(f"{exp}: confusion matrices");fig.tight_layout();save(fig,PLOTS/"confusion_matrices",f"{exp.lower()}_confusion_matrices")
    # Feature distributions and correlations from the real model inputs.
    for exp in comparison.experiment.unique():
        for variant in VARIANTS:
            path=OUT/exp/variant/"features.csv"
            if not path.exists():continue
            d=pd.read_csv(path);features=[c for c in d if c not in {"subject_id","window_index","original_label","stress_label"}]
            chosen=features[:min(12,len(features))]
            f,axes=plt.subplots(3,4,figsize=(14,8));
            for ax,col in zip(axes.ravel(),chosen):sns.boxplot(data=d,x="stress_label",y=col,ax=ax);ax.set_title(col,fontsize=8);ax.set_xlabel("class");ax.grid(alpha=.25)
            for ax in axes.ravel()[len(chosen):]:ax.set_axis_off()
            f.suptitle(f"{exp} {variant}: first extracted feature distributions by class");f.tight_layout();save(f,PLOTS/"feature_distributions"/exp,f"{variant}_feature_boxplots")
            corr=d[features].corr(method="pearson").iloc[:min(30,len(features)),:min(30,len(features))]
            f,ax=plt.subplots(figsize=(10,8));sns.heatmap(corr,cmap="vlag",center=0,ax=ax);ax.set(title=f"{exp} {variant}: Pearson feature correlation (first 30)");f.tight_layout();save(f,PLOTS/"correlation"/exp,f"{variant}_pearson")
    # Best all-signal logistic coefficients.
    best=comparison[comparison.experiment=="ALL_SIGNALS"].sort_values("f1",ascending=False).iloc[0];model=joblib.load(OUT/"ALL_SIGNALS"/best.variant/"model.joblib");features=pd.read_csv(OUT/"ALL_SIGNALS"/best.variant/"features.csv");cols=[c for c in features if c not in {"subject_id","window_index","original_label","stress_label"}];coef=model.named_steps["model"].coef_[0];order=np.argsort(np.abs(coef))[-20:]
    f,ax=plt.subplots(figsize=(10,7));ax.barh(np.array(cols)[order],coef[order]);ax.set(title=f"Feature importance: ALL_SIGNALS ({best.variant})",xlabel="Logistic-regression coefficient");ax.grid(axis="x",alpha=.3);save(f,PLOTS/"feature_importance","all_signals_coefficients")
    comparison.to_csv(PLOTS/"ml_results"/"model_performance.csv",index=False)

def label_plots(labels):
    kept=labels[np.isin(labels,[1,2,3])];binary=np.where(kept==2,"Stress","Non-stress");d=pd.Series(binary).value_counts().rename_axis("label").reset_index(name="samples")
    f,axes=plt.subplots(1,2,figsize=(10,4));sns.barplot(data=d,x="label",y="samples",ax=axes[0]);axes[0].set_title("Aligned label distribution");axes[1].pie(d.samples,labels=d.label,autopct="%1.1f%%");axes[1].set_title("Stress vs non-stress");
    for ax in axes:ax.grid(alpha=.25)
    f.tight_layout();save(f,PLOTS/"label_distribution","subject_s2_label_distribution");d.to_csv(PLOTS/"label_distribution"/"label_statistics.csv",index=False)

def latex_project():
    (REPORT/"chapters").mkdir(parents=True,exist_ok=True);(REPORT/"figures").mkdir(exist_ok=True);(REPORT/"tables").mkdir(exist_ok=True)
    shutil.copy2(PLOTS/"ml_results"/"overall_metric_comparison.png",REPORT/"figures"/"overall_metric_comparison.png");shutil.copy2(PLOTS/"signal_comparison"/"signalwise_f1.png",REPORT/"figures"/"signalwise_f1.png");shutil.copy2(PLOTS/"raw"/"E4_EDA"/"distribution_quality.png",REPORT/"figures"/"e4_eda_quality.png");shutil.copy2(PLOTS/"feature_importance"/"all_signals_coefficients.png",REPORT/"figures"/"feature_importance.png")
    perf=pd.read_csv(PLOTS/"ml_results"/"model_performance.csv");perf[perf.experiment=="ALL_SIGNALS"][["variant","accuracy","precision","recall","f1","roc_auc"]].to_latex(REPORT/"tables"/"overall_performance.tex",index=False,float_format="%.3f",escape=True)
    chapters={"introduction":"\\section{Introduction} This report documents a reproducible WESAD stress-detection analysis from physiological signals to subject-independent classification.","dataset":"\\section{Dataset and signals} WESAD provides synchronized wrist Empatica E4 and chest RespiBAN physiological measurements. Representative raw plots use subject S2; ML results use the completed three-subject experiment protocol.","preprocessing":"\\section{Preprocessing} Raw, filtered, and filtered-plus-normalized variants were compared while holding labels, five-second windows, participant split, and classifier fixed.","windowing":"\\section{Windowing} Signals were aligned to 4 Hz and windowed into five-second segments with 50\\% overlap. Window-level labels use the majority label.","feature_extraction":"\\section{Feature extraction} The experiment uses mean, median, standard deviation, variance, extrema, range, IQR, skewness, kurtosis, RMS, energy, entropy, MAV, AUC, slope, zero-crossing rate, and coefficient of variation per channel.","machine_learning":"\\section{Machine learning} Median imputation, standardization, and class-balanced logistic regression were evaluated with a held-out participant split.","results":"\\section{Results} Figure~\\ref{fig:overall} compares preprocessing variants. Table~\\ref{tab:overall} reports the all-signal results.\\input{tables/overall_performance.tex}","discussion":"\\section{Discussion} Interpret performance primarily through F1 and ROC-AUC because the held-out classes are imbalanced. The common 4-Hz branch is suitable for fair feature-table comparison but cannot preserve high-frequency ECG/EMG morphology.","conclusion":"\\section{Conclusion} The best pipeline and strongest signals are determined automatically from the generated comparison table; future work should use nested cross-validation and native-rate ECG/EMG features."}
    for name,text in chapters.items():(REPORT/"chapters"/f"{name}.tex").write_text(text+"\n",encoding="utf-8")
    main=r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}\usepackage{graphicx}\usepackage{booktabs}\usepackage{float}\usepackage{hyperref}
\title{WESAD Physiological Stress Detection: Methods and Results}\author{B.Tech Project}\date{\today}
\begin{document}\maketitle\tableofcontents
\input{chapters/introduction}\input{chapters/dataset}\input{chapters/preprocessing}\input{chapters/windowing}\input{chapters/feature_extraction}\input{chapters/machine_learning}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/overall_metric_comparison.png}\caption{Held-out all-signal metric comparison.}\label{fig:overall}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{figures/signalwise_f1.png}\caption{Signal-wise held-out F1 comparison.}\label{fig:signalwise}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{figures/e4_eda_quality.png}\caption{Representative E4 EDA distribution and quality checks.}\label{fig:eda}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=.9\linewidth]{figures/feature_importance.png}\caption{Largest standardized logistic-regression coefficients.}\label{fig:importance}\end{figure}
\input{chapters/results}\input{chapters/discussion}\input{chapters/conclusion}
\begin{thebibliography}{9}\bibitem{wesad} Schmidt et al. Introducing WESAD, ICMI 2018.\end{thebibliography}\end{document}"""
    (REPORT/"main.tex").write_text(main,encoding="utf-8")

def main():
    for d in ["raw","filtered","filtered_normalized","windowing","feature_extraction","feature_distributions","correlation","label_distribution","preprocessing_comparison","ml_results","signal_comparison","confusion_matrices","feature_importance"]:(PLOTS/d).mkdir(parents=True,exist_ok=True)
    data,labels=raw_signals();raw_plots(data);label_plots(labels);feature_and_ml_plots();latex_project()
    (PLOTS/"README.md").write_text("Generated PNG plots (320 dpi). Source: S2 raw record plus signalwise ML outputs.\n",encoding="utf-8")
    print(f"Created plots in {PLOTS} and LaTeX project in {REPORT}")
if __name__=="__main__":main()
