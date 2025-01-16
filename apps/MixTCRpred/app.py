import streamlit as st
import pandas as pd
from streamlit import session_state as ss
import os
from helpers.file_handling import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import subprocess
import atexit
import numpy as np
import sys


@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

def showSGR():
    singleron_base64 = read_image("src/singleron.png")
    with st.sidebar:
        st.markdown("---")
        st.markdown("This add is built by")
        st.markdown(
            f'<div style="margin-top: 0.25em; text-align: left;"><a href="https://singleron.bio/" target="_blank"><img src="data:image/png;base64,{singleron_base64}" alt="Homepage" style="object-fit: contain; max-width: 174px; max-height: 41px;"></a></div>',
            unsafe_allow_html=True,
        )

def binaryswitch(session_state, keys):
    for key in keys:
        if session_state[key] is True:
            session_state[key] = False
        else:
            session_state[key] = True

def get_sessionID():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime      = get_instance()
    session_id   = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session.id

### 画图函数
def plot4pMHC(df, auc_min, auc_max, num_bins):

    # 计算区间的宽度
    bin_width = (auc_max - auc_min) / num_bins

    # 计算区间的边界值
    bins = [auc_min + i * bin_width for i in range(num_bins)]
    bins.append(auc_max)  # 添加最大值作为区间的右边界

    # 使用pd.cut()函数将AUC值划分为5个区间，并为每个区间分配相应的标签
    labels        = [f"{bins[i]:.1f} - {bins[i+1]:.1f}" for i in range(num_bins)]
    df['AUC values'] = pd.cut(df['AUC_5fold'], bins=bins, labels=labels, right=False)
    fig           = px.scatter(df, x='Number_training_abTCR', y='AUC_5fold', color='AUC values', hover_name='MixTCRpred_model_name')
    fig.update_layout(xaxis_type="log")
    fig.update_layout(
        title="Model info with AUC values and trained TCR numbers",  # 设置图片标题
        title_x=0.3,
        xaxis = dict(
            title = 'log(TCR pretrained numbers)',
            tickfont=dict(size=16)  # 设置 x 轴标签的字体大小为 14
        ),
        yaxis = dict(
            title = 'Model AUC value',
            tickfont=dict(size=16)  # 设置 y 轴标签的字体大小为 14
        ),
        font=dict(  # 设置标题的字体大小
            size=16
        )
    )
    # 调整点的大小
    fig.update_traces(marker=dict(size=10))  # 设置点的大小为10
    return fig

def plot4result(df):
    df['names']   = "TRA:" + df['cdr3_TRA'] + "_" + "TRB:" + df['cdr3_TRB']
    df['sig']     = np.where(df['perc_rank'] < 0.01, "significant", "insignificant")
    fig           = px.scatter(df, x='score', y='perc_rank', color='sig', hover_name='names')

    fig.update_layout(
        title="Predictions with scores and ranks",  # 设置图片标题
        title_x=0.3,
        xaxis = dict(
            title = 'Score',
            tickfont=dict(size=16)  # 设置 x 轴标签的字体大小为 14
        ),
        yaxis = dict(
            title = 'Rank',
            tickfont=dict(size=16)  # 设置 y 轴标签的字体大小为 14
        ),
        font=dict(  # 设置标题的字体大小
            size=16
        )
    )
    # 调整点的大小
    fig.update_traces(marker=dict(size=10))  # 设置点的大小为10    
    return fig


#@st.cache_data
def readInfo(file):
    ss["df_info"] = pd.read_csv(file)

def main():
    # initial variables
    default_variables = {
        "df_info":None,
        "tcr_cutoff":0,
        "Host_species":None,
        "scTCR":None,
        "results":None,
    }
    if "df_info" not in ss:
        for key, value in default_variables.items():
            if key not in st.session_state:
                ss[key] = value

    ## 设置页面宽一些
    st.set_page_config(layout="wide")
    st.sidebar.markdown("# MixTCRpred")
    st.markdown("## Background")
    st.markdown(
        '''
        MixTCRpred is developed by [GfellerLab]((https://github.com/GfellerLab/MixTCRpred/tree/main)) and first published in [Nature Communications](https://www.nature.com/articles/s41467-024-47461-8).
        MixTCRpred accurately predicts T-cell receptors (TCRs) recognizing several viral and cancer epitopes (peptides displayed on MHC molecules or pMHCs). Predictions are available for 146 pMHCs. Accurate predictions were achieved for 43 pMHCs that have more than 50 training TCRs.
        ''')

    st.sidebar.markdown("## 1.MixTCRpred models")
    ss["Host_species"] = st.sidebar.selectbox(
        "Select which species to analyze:",
        ("All","HomoSapiens", "MusMusculus"),
        index=0, placeholder="Please select ..."
    )
    ss["tcr_cutoff"] = st.sidebar.number_input("Pretained TCR cutoff >=", 0)
    ss["auc_cutoff"] = st.sidebar.number_input("Modle AUC cutoff >=", 0.5, max_value=0.99)

    ## 加载模型的pMHC
    st.title("Pretrained models for the target pMHC")
    readInfo("./pretrained_models/info_models.csv")
    
    if ss["df_info"] is not None:
        df = pd.DataFrame()
        if ss["Host_species"] != "All":
            df = ss["df_info"][ ss["df_info"]["Host_species"] == ss["Host_species"] ]
        else:
            df = ss["df_info"]

        df = df[ df["Number_training_abTCR"] >= ss["tcr_cutoff"] ]
        df = df[ df["AUC_5fold"]             >= ss["auc_cutoff"] ]
        st.dataframe(df, hide_index=True)

        ## 展示分布
        fig_pMHC = plot4pMHC(df, ss["auc_cutoff"], 1, 5)
        st.plotly_chart(fig_pMHC, use_container_width=True)
        ## 选择模型
        pMHC_models      = ['A0201_GILGFVFTL', 'A0201_ELAGIGILTV']
        pMHC_models_sel  = st.selectbox(
            'Please select pretrained pMHC models:',
            pMHC_models,
            index=None
        )

    ## 输入scTCR data
    if pMHC_models_sel is None:
        st.stop()
    st.sidebar.markdown("## 2.Input TCR")
    input_selectbox = st.sidebar.selectbox(
        "What data would you like to use for analysis?",
        ("demo", "Upload new"),
        index=None, placeholder="Please select ..."
    )
    st.markdown("---")
    if input_selectbox == "demo":
        ss["scTCR"] = pd.read_csv("./test/test.csv")
    elif input_selectbox == "Upload new":
        st.subheader("Upload your TCR data (csv format, separated by commas). Please refer to the demo data for detials.")
        tcr_file    = st.file_uploader("Choose a csv file for TCR inputs.",   type="csv", disabled=True)
        ss["scTCR"] = pd.read_csv(tcr_file)

    if ss["scTCR"] is None:
        st.info('Next, please input TCR for analysis from the left slidebar!', icon="ℹ️")
        st.stop()
    else:
        st.title("Input TCRs for predictions:")
        df = ss["scTCR"]
        st.dataframe(df, hide_index=True)

        ## 按user id存放文件
        user_id   = get_sessionID()
        user_dir  = create_user_temp_dir(user_id)
        ## app重启后删掉相关文件夹
        atexit.register(cleanup_tmpdir, user_dir)
        tcr_csv   = os.path.join(user_dir, "tcr.csv")
        out_csv   = os.path.join(user_dir, "out.csv")
        ss["scTCR"].to_csv(tcr_csv, index=False)


    if st.button("Running MixTCRpred!"):
        ss['results']  = None
        python_path    = sys.executable
        #python MixTCRpred.py --model A0201_GILGFVFTL --input ./test/test.csv --output ./test/out_A0201_GILGFVFTL.csv
        #process1       = subprocess.Popen([python_path, "MixTCRpred.py", "--model", pMHC_models_sel, "--input", tcr_csv, "--output", out_csv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #return_code    = process1.wait()
        #stdout, stderr = process1.communicate()
        cmd_line     = " ".join([python_path, "MixTCRpred.py", "--model", pMHC_models_sel, "--input", tcr_csv, "--output", out_csv])
        return_code  = os.system(cmd_line)
        if return_code == 0:
            st.write('MixTCRpred executed successfully!')
            ss['results'] = pd.read_csv(out_csv, comment='#')
        else:
            st.write('MixTCRpred runnning failed!')
            st.write("please check your data !")
            st.stop()


    ## 展示结果
    if ss['results'] is not None:
        st.markdown("---")
        st.sidebar.markdown("## 3.Results")
        st.title("Results:")
        df = ss['results']
        
        st.dataframe(df, hide_index=True)
        csv = convert_df(ss['results'])
        st.download_button(
           "Download tables",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )

        ## 以score和rank展示结果
        fig_result = plot4result(df)
        st.plotly_chart(fig_result, use_container_width=True)





    ## 展示公司和引用信息
    st.sidebar.markdown("## 4.Citations")
    showSGR()
    st.markdown("---")
    st.markdown(
        '''
        ### Citations:
         * [MixTCRpred](https://github.com/GfellerLab/MixTCRpred/tree/main): Croce G, Bobisse S, Moreno D L, et al. Deep learning predictions of TCR-epitope interactions reveal epitope-specific chains in dual alpha T cells[J]. Nature Communications, 2024, 15(1): 3211.
         * Pétremand R, Chiffelle J, Bobisse S, et al. Identification of clinically relevant T cell receptors for personalized T cell therapy using combinatorial algorithms[J]. Nature Biotechnology, 2024: 1-6.
        '''
    )


if __name__ == "__main__":
    main()
