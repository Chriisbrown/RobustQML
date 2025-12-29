# Import from other modules
from data.dataset import DataSet

# SELECTED = {
#     "DY": "DY to ll",
#     "QCD": "QCD inclusive",
#     "SingleHiggs": "VBFHtautau",
#     "top": "tt all-lept",
#     "diboson": "WZ (semi-leptonic)",
#     "diHiggs": "HH bbtautau",
# }

SELECTED = {
    # "DY": "DY to ll",
    # "QCD": "QCD inclusive",
    # "QCDbb": "QCD bb",
    # "Minbias": "Minbias / Soft QCD",
    # "TTallh" : "tt all-hadr",
    # "HH4b"  : "HH 4b",
    # "TTsemil" : "tt semi-lept",
    # "TTalll" : "tt all-lept",
    # "ttH" : "ttH incl",
    # "ttW" : "ttW incl",
    # "ttZ" : "ttZ incl",
    # "Wqq" : "W -> qq",
    # "Wlv" : "W -> lv",
    # "Zjets" : "Z -> vv + jet","
    # "Zqq" : "Z -> qq (uds)",
    # "Zbb" : "Z -> bb",
    # "Zcc" : "Z -> cc",
    "upsilonl" : "upsilon -> l" ,
    "VBFHcc" : "VBFHcc" ,
    "ggHbb" : "ggHbb"
}

PROCESS_TO_FOLDER = {
    "upsilon -> l": "upsilon_to_leptons",
    
    # DY / Z / W
    "DY to ll": "DYJetsToLL_13TeV-madgraphMLM-pythia8",
    "Z -> vv + jet": "ZJetsTovv_13TeV-madgraphMLM-pythia8",
    "Z -> qq (uds)": "ZJetsToQQ_13TeV-madgraphMLM-pythia8",
    "Z -> bb": "ZJetsTobb_13TeV-madgraphMLM-pythia8",
    "Z -> cc": "ZJetsTocc_13TeV-madgraphMLM-pythia8",
    "W -> lv": "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    "W -> qq": "WJetsToQQ_13TeV-madgraphMLM-pythia8",
    "gamma": "gamma",
    "gamma + V": "gamma_V",
    "tri-gamma": "tri_gamma",

    # QCD
    "QCD inclusive": "QCD_HT50toInf",
    "QCD bb": "QCD_HT50tobb",
    "Minbias / Soft QCD": "minbias",

    # top
    "tt all-hadr": "tt0123j_5f_ckm_LO_MLM_hadronic",
    "tt semi-lept": "tt0123j_5f_ckm_LO_MLM_semiLeptonic",
    "tt all-lept": "tt0123j_5f_ckm_LO_MLM_leptonic",
    "ttH incl": "ttH_incl",
    "tttt": "tttt_incl",
    "ttW incl": "ttW_incl",
    "ttZ incl": "ttZ_incl",

    # dibosons
    "WW (all-leptonic)": "WW_leptonic",
    "WW (all-hadronic)": "WW_hadronic",
    "WW (semi-leptonic)": "WW_semileptonic",
    "WZ (all-leptonic)": "WZ_leptonic",
    "WZ (all-hadronic)": "WZ_hadronic",
    "WZ (semi-leptonic)": "WZ_semileptonic",
    "ZZ (all-leptonic)": "ZZ_leptonic",
    "ZZ (all-hadronic)": "ZZ_hadronic",
    "ZZ (semi-leptonic)": "ZZ_semileptonic",
    "VVV": "VVV_incl",
    "VH incl": "VH_incl",

    # single-Higgs
    "ggHbb": "ggHbb",
    "ggHcc": "ggHcc",
    "ggHgammagamma": "ggHgammagamma",
    "ggHgluglu": "ggHgluglu",
    "ggHtautau": "ggHtautau",
    "ggHWW": "ggHWW",
    "ggHZZ": "ggHZZ",
    "VBFHbb": "VBFHbb",
    "VBFHcc": "VBFHcc",
    "VBFHgammagamma": "VBFHgammagamma",
    "VBFHgluglu": "VBFHgluglu",
    "VBFHtautau": "VBFHtautau",
    "VBFHWW": "VBFHWW",
    "VBFHZZ": "VBFHZZ",

    # di-Higgs
    "HH 4b": "HH_4b",
    "HH bbtautau": "HH_bbtautau",
    "HH bbWW": "HH_bbWW",
    "HH bbZZ": "HH_bbZZ",
    "HH bbgammagamma": "HH_bbgammagamma",
}

CLASS_NAMES = list(SELECTED.keys())
PRETTY = {c: SELECTED[c] for c in CLASS_NAMES}
FOLDER = {c: PROCESS_TO_FOLDER[PRETTY[c]] for c in CLASS_NAMES}
print("Classes:", CLASS_NAMES)
print("Pretty per class:", PRETTY)
print("Folder per class:", FOLDER)

for class_name in CLASS_NAMES:
    print(PROCESS_TO_FOLDER[PRETTY[class_name]])
    data_set = DataSet.fromHF(PROCESS_TO_FOLDER[PRETTY[class_name]])
    data_set.pretty_name = PRETTY[class_name]
    data_set.save_h5('dataset/'+class_name)
    data_set.plot_inputs('dataset/'+class_name)