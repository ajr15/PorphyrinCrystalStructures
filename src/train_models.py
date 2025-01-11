from sqlalchemy import text as sqltext
import random
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import pandas as pd
from openbabel import openbabel as ob
from featurizers import StructurePropertyFeaturizer, SubstituentPropertyFeaturizer, FunctionFeaturizer, Featurizer
from read_to_sql import Substituent, StructureProperty
import utils

def metal_radius(session, sid: int):
    """Get the VDW radius of the metal center"""
    metal = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    metal = utils.mol_from_smiles(metal).GetAtom(1)
    return ob.GetVdwRad(metal.GetAtomicNum())

def _non_planarity_helper(session, sid, mode, units):
    v = session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == mode).filter(StructureProperty.units == units).all()
    return abs(v[0][0])

def non_planarity(mode, units):
    s = mode if "total" in mode else mode + " non planarity"
    return lambda session, sid: _non_planarity_helper(session, sid, s, units)

def _dominant_mode_helper(session, sid, th, mode):
    structprops = session.query(StructureProperty).\
                filter(StructureProperty.structure == sid).\
                filter(StructureProperty.property.contains("non planarity")).\
                filter(StructureProperty.units == "%").all()
    total_np = _non_planarity_helper(session, sid, "total out of plane (exp)", "A")
    max_mode = structprops[0]
    for sp in structprops[1:]:
        if max_mode.value < sp.value:
            max_mode = sp
    if total_np >= th:
        return max_mode.property.split()[0] == mode
    else:
        return False

def dominant_mode(mode: str, th: float=1):
    """featurizer to check the dominant mode of a molecule"""
    return lambda session, sid: _dominant_mode_helper(session, sid, th, mode)

MACROCYCLE_POSITIONS = ["meso1", "beta1", "beta2", "meso2", "beta3", "beta4", "meso3", "beta5", "beta6", "meso4", "beta7", "beta8"]
MACROCYCLE_POSITIONS = [MACROCYCLE_POSITIONS[(3 * i):] + MACROCYCLE_POSITIONS[:(3 * i)] for i in range(4)]
AXIAL_POSITIONS = [["axial1", "axial2"], ["axial2", "axial1"]]
STYPE = "porphyrin"

def reduced_distances_helper(pname, session, sid):
    feat = SubstituentPropertyFeaturizer(pname, None, MACROCYCLE_POSITIONS[0], navalue=None)
    df = feat.featurize(session, [sid])
    df["beta-beta"] = np.mean(df[["beta1", "beta3", "beta5", "beta7"]].values)
    df["beta-meso"] = np.mean(df[["meso" + str(i + 1) for i in range(4)] + ["beta2", "beta4", "beta6", "beta8"]].values)
    return df[["beta-beta", "beta-meso"]].values.tolist()[0]

def reduced_distances(pname):
    return lambda session, sid: reduced_distances_helper(pname, session, sid)

def reduced_cone_angles(session, sid):
    feat = SubstituentPropertyFeaturizer("cone angle", None, MACROCYCLE_POSITIONS[0], navalue=-1)
    df = feat.featurize(session, [sid])
    df["beta"] = np.mean(df[[c for c in df.columns if "beta" in c]].values)
    df["meso"] = np.mean(df[[c for c in df.columns if "meso" in c]].values)
    return df[["beta", "meso"]].values.tolist()[0]

def axial_features(session, sid):
    feat = SubstituentPropertyFeaturizer("cone angle", None, AXIAL_POSITIONS[0], navalue=-1)
    df = feat.featurize(session, [sid])
    empty_spots = np.sum(df.eq(-1).values)
    axial_angles = df[~df.eq(-1)].dropna(axis=1).values[0]
    mean_angle = np.mean(axial_angles) if len(axial_angles) > 0 else None
    return [6 - empty_spots, mean_angle]

def avg_pyrrole_homa(session, sid):
    props = ['pyrrole1 homa', 'pyrrole2 homa', 'pyrrole3 homa', 'pyrrole4 homa']
    units = [None for _ in range(len(props))]
    feat = StructurePropertyFeaturizer(props, units, navalue=None)
    return np.mean(feat.featurize(session, [sid]).values)

def mixed_angle_distance(session, sid):
    feat = SubstituentPropertyFeaturizer("covalent nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) +\
            SubstituentPropertyFeaturizer("cone angle", None, MACROCYCLE_POSITIONS[0], navalue=-1)
    return feat.featurize(session, [sid]).values


MEATL_AXIAL_FEATURES = FunctionFeaturizer(["coordination", "axial_angle"], axial_features, -1) + FunctionFeaturizer("metal_radius", metal_radius, navalue=None)

FEATURIZERS = {
    # "cone_angles": SubstituentPropertyFeaturizer("cone angle", None, MACROCYCLE_POSITIONS[0], navalue=-1) + MEATL_AXIAL_FEATURES,
    # "vdw_distances": SubstituentPropertyFeaturizer("vdw nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) + MEATL_AXIAL_FEATURES,
    # "covalent_distances": SubstituentPropertyFeaturizer("covalent nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) + MEATL_AXIAL_FEATURES,
    # "nn_distances": SubstituentPropertyFeaturizer("None nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) + MEATL_AXIAL_FEATURES,
    "reduced_vdw_distances": FunctionFeaturizer(["beta-beta", "beta-meso"], reduced_distances("vdw nn dist"), None) + MEATL_AXIAL_FEATURES,
    "reduced_cone_angles": FunctionFeaturizer(["beta_angle", "meso_angle"], reduced_cone_angles, None) + MEATL_AXIAL_FEATURES,
    # "angles_and_distances": FunctionFeaturizer(["beta_angle", "meso_angle"], reduced_cone_angles, None) + FunctionFeaturizer(["beta-beta", "beta-meso"], reduced_distances("vdw nn dist"), None) + MEATL_AXIAL_FEATURES,
}



REGRESSION_TARGETS = {
    # "outer_homa": StructurePropertyFeaturizer(["outer_circuit homa"], [None], navalue=None),
    "inner_homa": StructurePropertyFeaturizer(["inner_circuit homa"], [None], navalue=None),
    "pyrrole_homa": FunctionFeaturizer("pyrrole homa", avg_pyrrole_homa, navalue=None),
    "total_out_of_plane": FunctionFeaturizer("total out of plane", non_planarity("total out of plane (exp)", "A"), navalue=None),
    "abs_ruffling": FunctionFeaturizer("abs. ruffling", non_planarity("ruffling", "A"), navalue=None),
    "abs_saddling": FunctionFeaturizer("abs. saddling", non_planarity("saddling", "A"), navalue=None),
    # "abs_doming": FunctionFeaturizer("abs. doming", non_planarity("doming", "A"), navalue=None),
}

CLASSIFICATION_TARGETS = {
    "saddling": FunctionFeaturizer("saddling", dominant_mode("saddling"), navalue=None),
    "ruffling": FunctionFeaturizer("ruffling", dominant_mode("ruffling"), navalue=None),
    "doming": FunctionFeaturizer("doming", dominant_mode("doming"), navalue=None),
}

MODELS = {
    "rf": (RandomForestRegressor, {"n_estimators": 1000}),
    "lasso": (LassoCV, {"n_alphas": 10000}),
}


def augment_data(X, y):
    new_X = []
    new_y = []
    for macro_pos in MACROCYCLE_POSITIONS:
        for i in range(len(y)):
            idxs = macro_pos + X.columns[len(macro_pos):].tolist()
            equiv = X.iloc[i, :].loc[idxs].to_numpy()
            new_X.append(equiv)
            new_y.append(y.iloc[i, :])
    return pd.DataFrame(new_X), pd.DataFrame(new_y)

def get_unique_structures(session, position, all_sids):
    """Get IDs of structures containing at least on unique substituent on a given position"""
    q = "substituent FROM (SELECT substituent, COUNT(*) as nstructs FROM (SELECT structure, substituent FROM substituents WHERE position=\"$POSITION$\" GROUP BY substituent, structure) GROUP BY substituent) WHERE nstructs=1"
    res = session.query(sqltext(q.replace("$POSITION$", position))).all()
    q = "structure FROM substituents WHERE substituent=\"$SUBS$\" AND position=\"$POS$\""
    ajr = []
    for sub in res:
        sid = session.query(sqltext(q.replace("$SUBS$", sub[0]).replace("$POS$", position))).all()[0][0]
        if sid in all_sids:
            ajr.append(sid)
    return ajr

def train_test_split(session, n: int):
    """Method to extract a more robust train-test split. makes sure that test set elements contain at least one substituent different from the train set."""
    all_sids = utils.sids_by_type(session, stype="porphyrin")
    unique_structures = list(set(get_unique_structures(session, "beta", all_sids) + get_unique_structures(session, "meso", all_sids) + get_unique_structures(session, "axial", all_sids)))
    selected = []
    for _ in range(n):
        s = np.random.choice(unique_structures, 1)[0]
        selected.append(s)
        all_sids.remove(s)
        unique_structures.remove(s)
    return all_sids, selected

def make_data(session, featurizer: Featurizer, target: Featurizer, test_size: int=30, augment: bool=False):
    sids = utils.sids_by_type(session, STYPE)
    train_sids, test_sids = train_test_split(session, n=test_size)
    X = featurizer.featurize(session, sids)
    y = target.featurize(session, sids)
    xtrain = X.loc[train_sids, :].values
    ytrain = y.loc[train_sids, :].values
    xtest = X.loc[test_sids, :].values
    ytest = y.loc[test_sids, :].values
    if augment:
        xtrain, ytrain = augment_data(xtrain, ytrain)
        xtest, ytest = augment_data(xtest, ytest)
    return xtrain, xtest, ytrain, ytest


def run_bootstraps(session, modelargs, featurizer: Featurizer, target: Featurizer, test_size: int, augment: bool, n_bootstraps: int):
    data = []
    models = []
    for _ in range(n_bootstraps):
        xtrain, xtest, ytrain, ytest = make_data(session, featurizer, target, test_size, augment)
        nxtrain = utils.normalize(xtrain, xtrain)
        nxtest = utils.normalize(xtest, xtrain)
        nytrain = utils.normalize(ytrain, ytrain)
        model = modelargs[0](**modelargs[1])
        model.fit(nxtrain, nytrain)
        ytrain_pred = utils.unnormalize(model.predict(nxtrain), ytrain)
        ytest_pred = utils.unnormalize(model.predict(nxtest), ytrain)
        # calculating metrics
        metrics = utils.estimate_regression_fit(ytrain_pred, ytrain, "train_")
        metrics.update(utils.estimate_regression_fit(ytest_pred, ytest, "test_"))
        # adding to stack
        models.append(model)
        data.append(metrics)
    return models, pd.DataFrame(data)


def main(session, models_dir: str, augment: bool):
    targets = REGRESSION_TARGETS
    for feat in FEATURIZERS:
        ajr = {}
        for target in targets:
            for model in MODELS:
                print("RUNNING {} WITH {} AND {} MODEL".format(feat, target, model))
                # fixing random seed
                np.random.seed(0)
                random.seed(0)
                # running fit
                models, df = run_bootstraps(session, MODELS[model], FEATURIZERS[feat], targets[target], n_bootstraps=10, augment=augment, test_size=30)
                # saving results
                path = os.path.join(models_dir, "{}_{}_{}".format(feat, target, model))
                if not os.path.isdir(path):
                    os.mkdir(path)
                # save dataframe
                df.to_csv(os.path.join(path, "metrics.csv"))
                # save models
                for i, model in enumerate(models):
                    ajr = os.path.join(path, str(i))
                    if not os.path.isdir(ajr):
                        os.mkdir(ajr)
                    utils.save_model(model, os.path.join(path, str(i)))

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from featurizers import ComboFeaturizer
    utils.define_pallet()
    engine = create_engine("sqlite:///{}".format(os.environ["CRYSTAL_MAIN_DB"]))
    session = sessionmaker(bind=engine)()
    main(session, os.environ["CRYSTAL_SRC_DIR"] + "/models", False)
