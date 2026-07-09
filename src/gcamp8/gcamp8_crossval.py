from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rslds.rSLDS import cross_val, DataType
from src.gcamp8.gcamp8_load_util import session_concat_pipeline

if __name__ == "__main__":
    
    folder = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2"
    plot_key_go = "#23819M_GCaMP8m_rg_PoM-FOV2/session_concat_crossval/go"
    plot_key_nogo = "#23819M_GCaMP8m_rg_PoM-FOV2/session_concat_crossval/nogo"
    plot_key_full = "#23819M_GCaMP8m_rg_PoM-FOV2/session_concat_crossval/full"

    date1 = "050425" # NAIVE
    date2 = "050625" # INTERMEDIATE
    date3 = "050825" # INTERMEDIATE
    date4 = "051325" # EXPERT
    date5 = "051925" # EXPERT

    dates = [date1, date2, date3, date4, date5]

    go_concat = session_concat_pipeline(folder, dates, trial_selection="go", path_type="sliceTCA")
    nogo_concat = session_concat_pipeline(folder, dates, trial_selection="nogo", path_type="sliceTCA")
    full_concat = session_concat_pipeline(folder, dates, path_type="sliceTCA")

    dims = [9, 10, 11, 12, 13, 14]
    states = [9, 10, 11, 12, 13]

    cross_val(full_concat, plot_key_full, dims, states, type=DataType.GCaMP8)

    cross_val(go_concat, plot_key_go, dims, states, type=DataType.GCaMP8)

    cross_val(nogo_concat, plot_key_nogo, dims, states, type=DataType.GCaMP8)



