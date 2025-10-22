
import os
import cv2
from tqdm import tqdm
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

class TestMetric(object):
    def __init__(self, root_preds_path, gt_path):
        self.MAE = MAE()
        self.FM = Fmeasure()
        self.SM = Smeasure()
        self.EM = Emeasure()
        self.WFM = WeightedFmeasure()

        self.root_preds_path = root_preds_path
        self.gt_path = gt_path

    def back_results(self):
        data_root = "./test_data"
        mask_root = self.gt_path
        pred_root = self.root_preds_path

        mask_name_list = sorted(os.listdir(mask_root))
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            self.FM.step(pred=pred, gt=mask)
            self.WFM.step(pred=pred, gt=mask)
            self.SM.step(pred=pred, gt=mask)
            self.EM.step(pred=pred, gt=mask)
            self.MAE.step(pred=pred, gt=mask)

        fm = self.FM.get_results()["fm"]
        wfm = self.WFM.get_results()["wfm"]
        sm = self.SM.get_results()["sm"]
        em = self.EM.get_results()["em"]
        mae = self.MAE.get_results()["mae"]

        results = {
            "maxFm": round(fm["curve"].max(), 3),
            "wFmeasure": round(wfm, 3),
            "MAE": round(mae, 3),
            "Smeasure": round(sm, 3),
            "meanEm": round(em["curve"].mean(), 3),
            "meanFm": round(fm["curve"].mean(), 3)
        }
        return results

if __name__ == "__main__":
    # Open a file to write the results
    with open("m.txt", "w") as file:


        print("47-VD")
        file.write("47-VD\n")
        root_preds_path = r"..."
        gt_path = r"..."
        testMetric = TestMetric(root_preds_path, gt_path)
        results = testMetric.back_results()
        print(results)
        file.write(f"{results}\n")



    
    file.flush()
    file.close()