import pandas as pd

def read_excel(file_path, sheet_name=0):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print("Error reading Excel file:", e)
        return None

def main():
    # Example Excel file path
    file_path = "Mod-USWorldRAnking.ods"

    # Read the Excel file
    df = read_excel(file_path)
    if df is not None:
        rank = df['Column1'].tolist()
        schoolName = df['Column2'].tolist()
        state = df['Column3'].tolist()
        pubPri = df['Column4'].tolist()
        prevrank = df['Column5'].tolist()
        change = df['Column6'].tolist()
        citperpub = df['Column7'].tolist()
        fieldwgtcitImp = df['Column8'].tolist()
        pubCited1 = df['Column9'].tolist()
        pubCited2 = df['Column10'].tolist()
        totpub = df['Column11'].tolist()
        res_exp = df['Column12'].tolist()
        res_expr_fac = df['Column13'].tolist()
        peer_ass = df['Column14'].tolist()
        req_ass = df['Column15'].tolist()
        doctoral_deg_granted = df['Column16'].tolist()
        acceptance = df['Column17'].tolist()
        per_fac = df['Column18'].tolist()
        docStudent = df['Column19'].tolist()
        score = df['Column20'].tolist()

        # Print the vectors
        print("Vector from Column1:", column1)
        print("Vector from Column2:", column2)
        print("Vector from Column3:", column3)
        print("Vector from Column4:", column4)
        print("Vector from Column5:", column5)
        print("Vector from Column6:", column6)
        print("Vector from Column7:", column7)
        print("Vector from Column8:", column8)
        print("Vector from Column9:", column9)
        print("Vector from Column10:", column10)
        print("Vector from Column11:", column11)
        print("Vector from Column12:", column12)
        print("Vector from Column13:", column13)
        print("Vector from Column14:", column14)
        print("Vector from Column15:", column15)
        print("Vector from Column16:", column16)
        print("Vector from Column17:", column17)
        print("Vector from Column18:", column18)
        print("Vector from Column19:", column19)
        print("Vector from Column20:", column20)


if __name__ == "__main__":
    main()

