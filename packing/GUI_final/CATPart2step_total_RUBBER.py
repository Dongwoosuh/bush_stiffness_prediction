import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("..\\pycatia"))

##########################################################################

def convert_catpart_to_step(work_dir: str):

    from pycatia import catia
    from pycatia.mec_mod_interfaces.part_document import PartDocument

    # catpart_file_root = Path(work_dir) / "Input_CATpart"
    catpart_file_root = Path(work_dir)

    if not catpart_file_root.exists():
        print(f"{catpart_file_root} 경로가 존재하지 않습니다.")
        return ValueError("Cartpart Folder path does not exist")

    # Check if there are any .CATPart files in the directory
    catpart_files = list(catpart_file_root.glob("*.CATPart"))
    if not catpart_files:
        print(f"{catpart_file_root} 폴더에 .CATPart 파일이 존재하지 않습니다.")
        raise ValueError("No .CATPart files found in the specified folder.")

    for catpart_file in catpart_file_root.glob("*.CATPart"):
        catpart_file_name = catpart_file.stem
        print(f"\nProcessing file: {catpart_file_name}.CATPart")
        
        # Set output folder for current file.
        # output_folder = Path(f"{work_dir}/Input_AI/{catpart_file_name}")
        input_ai_folder = Path(f"{work_dir}/..")
        output_folder = Path(f"{input_ai_folder}/Input_AI/{catpart_file_name}")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        caa = catia()
        documents = caa.documents
        part_document: PartDocument = documents.open(str(catpart_file))
        assert isinstance(part_document, PartDocument)
        
        part = part_document.part
        num_bodies = part.bodies.count
        # print("Total number of bodies:", num_bodies)
        
        source_selection = part_document.selection
        body_counter = 1
        
        # Loop through all bodies (skip bodies with names starting with '#')
        for i in range(1, num_bodies + 1):
            body = part.bodies.item(i)
            if body.name.startswith("#"):
                continue

            # print(f"Body: {body.name}")

            source_selection.clear()
            source_selection.add(body)
            source_selection.copy()

            new_part_doc: PartDocument = documents.add("Part")
            new_part = new_part_doc.part

            new_selection = new_part_doc.selection
            new_selection.clear()
            try:
                target_body = new_part.bodies.item(1)
                new_selection.add(target_body)
            except Exception as e:
                print(f"Error accessing default body of new document: {e}")

            try:
                new_selection.paste()
            except Exception as e:
                print(f"Paste failed: {e}")

            new_part.update()

            output_step_file = Path(f"{catpart_file_name}_Part_{body_counter}.stp")
            output_step_file_path = output_folder / output_step_file
            try:
                new_part_doc.export_data(str(output_step_file_path), "stp", overwrite=True)
                print(f"\t >> Export successful for {body.name}: {output_step_file}")
            except Exception as e:
                print(f"\t >> Export failed for {body.name}: {e}")
                
            # If the body name includes "rubber" (case-insensitive),
            # and additionally if "stopper" or "bridge" are in the name,
            # export with the respective file name.
            if "rubber" in body.name.lower():
                # 기본 파일명 (rubber 포함)
                rubber_filename = f"{catpart_file_name}_RUBBER.stp"
                if "stopper" in body.name.lower():
                    rubber_filename = f"{catpart_file_name}_RUBBER_STOPPER.stp"
                elif "bridge" in body.name.lower():
                    rubber_filename = f"{catpart_file_name}_RUBBER_BRIDGE.stp"
                
                output_rubber_file = Path(rubber_filename)
                output_rubber_file_path = output_folder / output_rubber_file
                try:
                    new_part_doc.export_data(str(output_rubber_file_path), "stp", overwrite=True)
                    print(f"\t >> Rubber export successful for {body.name}: {output_rubber_file}")
                except Exception as e:
                    print(f"\t >> Rubber export failed for {body.name}: {e}")

            new_part_doc.close()
            body_counter += 1
        
        part_document.close()

# CLI로 실행할 때만
if __name__ == "__main__":

    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True, help="경로 지정: 예) C:/mydir")
    args = parser.parse_args()

    convert_catpart_to_step(args.work_dir)
    print(f"[CATPart to Step Converter] Conversion complete")