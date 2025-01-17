import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import threading

from src.methods import hamming_code as hmc
from src.methods import bch_code_dwt as bch_dwt
from src.methods import dtc_psychovisual as dct_psy


intro_text = (
    "Tato aplikace umožňuje skrývání zpráv do videí pomocí tří různých metod:\n\n\n"
    "1. LSB s Hamming kódem (7,4) - využívá nejméně významné bity pro skrytí dat a opravuje chyby pomocí Hamming kódu.\n\n"
    "2. DWT s BCH kódy (15, 11) - kombinuje diskrétní vlnkovou transformaci s BCH kódy pro opravu chyb.\n\n"
    "3. DCT s psychovisuální analýzou - používá diskrétní kosinovou transformaci a psychovizuální model pro určení vhodného místa pro vkládání. Tato metoda je vhodná jen pokud video obsahuje pohybující se objekty.\n\n\n"
    "Vyberte prosím možnost 'Zakódovat' pro skrytí zprávy nebo 'Dekódovat' pro odhalení skryté zprávy."
)


class LoadingWindow:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Zpracování...")
        self.top.geometry("300x100")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()

        self.progress = ttk.Progressbar(
            self.top, mode="indeterminate", length=250)
        self.progress.pack(pady=20)

        self.label = tk.Label(
            self.top, text="Probíhá zpracování, prosím čekejte...")
        self.label.pack()

        self.progress.start()

    def close(self):
        self.progress.stop()
        self.top.destroy()


class SteganographyApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Video steganografie")
        self.master.geometry("800x700")

        self.methods = [
            "LSB - Hamming code (7,4)", "DWT - BCH codes", "DCT psychovisual and object motion"]
        self.main_menu()

    def main_menu(self):
        self.clear_window()

        main_frame = tk.Frame(self.master)
        main_frame.pack(expand=True)

        tk.Label(main_frame, text="Video steganografie",
                 font=("Arial", 24)).pack(pady=40)

        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        intro_text_widget = tk.Text(text_frame, font=(
            "Arial", 14), wrap=tk.WORD, padx=10, pady=10, height=10)
        intro_text_widget.pack(pady=10)
        intro_text_widget.insert(tk.END, intro_text)
        intro_text_widget.config(state=tk.DISABLED)

        button_frame = tk.Frame(main_frame)
        button_frame.pack()

        tk.Button(button_frame, text="Zakódovat", command=self.encode_menu, font=(
            "Arial", 16), width=10).pack(side=tk.LEFT, padx=20)
        tk.Button(button_frame, text="Dekódovat", command=self.decode_menu, font=(
            "Arial", 16), width=10).pack(side=tk.LEFT, padx=20)

    def encode_menu(self):
        self.clear_window()

        main_frame = tk.Frame(self.master)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.selected_method = tk.StringVar(main_frame)
        self.selected_method.set(self.methods[0])

        method_menu = ttk.OptionMenu(main_frame, self.selected_method,
                                     self.methods[0], *self.methods, command=self.update_encode_options)
        method_menu.pack(pady=10)

        # Frame for selecting video
        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.X, pady=10)

        tk.Button(video_frame, text="Vybrat video", command=self.select_video, font=(
            "Arial", 14)).pack(side=tk.RIGHT)
        self.video_path_var = tk.StringVar()
        tk.Entry(video_frame, textvariable=self.video_path_var, font=(
            "Arial", 12), width=100, state='readonly').pack(side=tk.LEFT, padx=10)

        tk.Label(main_frame, text="Zpráva:", font=("Arial", 14)).pack()

        # Text widget with scrollbar for message
        message_frame = tk.Frame(main_frame)
        message_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.message_text = tk.Text(message_frame, font=(
            "Arial", 14), height=10, wrap=tk.WORD)
        self.message_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        message_scrollbar = tk.Scrollbar(message_frame)
        message_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.message_text.config(yscrollcommand=message_scrollbar.set)
        message_scrollbar.config(command=self.message_text.yview)

        self.additional_options_frame = tk.Frame(main_frame)
        self.additional_options_frame.pack(fill=tk.X, pady=10)

        self.update_encode_options(self.methods[0])

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Zpět", command=self.main_menu,
                  font=("Arial", 14)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Potvrdit", command=self.encode,
                  font=("Arial", 14)).pack(side=tk.LEFT, padx=10)

    def update_encode_options(self, selected_method):
        for widget in self.additional_options_frame.winfo_children():
            widget.destroy()

        validate_numeric = self.master.register(self.is_numeric)

        if selected_method == "LSB - Hamming code (7,4)":
            tk.Label(self.additional_options_frame,
                     text="Klíč 1:", font=("Arial", 14)).pack()
            self.key1_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key1_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 2:", font=("Arial", 14)).pack()
            self.key2_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key2_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 3:", font=("Arial", 14)).pack()
            self.key3_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key3_entry.pack()

        elif selected_method == "DWT - BCH codes":
            tk.Label(self.additional_options_frame,
                     text="Klíč 1:", font=("Arial", 14)).pack()
            self.col_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.col_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 2:", font=("Arial", 14)).pack()
            self.row_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.row_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="BCH Code:", font=("Arial", 14)).pack()
            self.bch_code_var = tk.StringVar()
            self.bch_code_combobox = ttk.Combobox(self.additional_options_frame, textvariable=self.bch_code_var, font=(
                "Arial", 14), values=[5, 7, 11], state='readonly')
            self.bch_code_combobox.pack()

        elif selected_method == "DCT psychovisual and object motion":
            """# Frame for selecting motion blocks file
            motion_blocks_frame = tk.Frame(self.additional_options_frame)
            motion_blocks_frame.pack(fill=tk.X, pady=10)

            tk.Button(motion_blocks_frame, text="Vybrat soubor JSON", command=self.select_motion_blocks_file, font=("Arial", 14)).pack(side=tk.RIGHT)
            self.motion_blocks_path_var = tk.StringVar()
            tk.Entry(motion_blocks_frame, textvariable=self.motion_blocks_path_var, font=("Arial", 12), width=100, state='readonly').pack(side=tk.LEFT, padx=10)
            """

    def decode_menu(self):
        self.clear_window()

        main_frame = tk.Frame(self.master)
        main_frame.pack(expand=True)

        self.selected_method = tk.StringVar(main_frame)
        self.selected_method.set(self.methods[0])

        method_menu = ttk.OptionMenu(main_frame, self.selected_method,
                                     self.methods[0], *self.methods, command=self.update_decode_options)
        method_menu.pack(pady=10)

        # Frame for selecting video
        video_frame = tk.Frame(main_frame)
        video_frame.pack(fill=tk.X, pady=10)

        tk.Button(video_frame, text="Vybrat stego-video",
                  command=self.select_video, font=("Arial", 14)).pack(side=tk.RIGHT)
        self.video_path_var = tk.StringVar()
        tk.Entry(video_frame, textvariable=self.video_path_var, font=(
            "Arial", 12), width=100, state='readonly').pack(side=tk.LEFT, padx=10)

        self.additional_options_frame = tk.Frame(main_frame)
        self.additional_options_frame.pack(fill=tk.X, pady=10)

        self.update_decode_options(self.methods[0])

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Zpět", command=self.main_menu,
                  font=("Arial", 14)).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Potvrdit", command=self.decode,
                  font=("Arial", 14)).pack(side=tk.LEFT, padx=10)

    def update_decode_options(self, selected_method):
        for widget in self.additional_options_frame.winfo_children():
            widget.destroy()

        validate_numeric = self.master.register(self.is_numeric)

        if selected_method == "LSB - Hamming code (7,4)":
            tk.Label(self.additional_options_frame,
                     text="XOR Klíč (7 bitů):", font=("Arial", 14)).pack()
            self.xor_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.xor_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 1:", font=("Arial", 14)).pack()
            self.key1_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key1_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 2:", font=("Arial", 14)).pack()
            self.key2_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key2_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 3:", font=("Arial", 14)).pack()
            self.key3_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.key3_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Délka zprávy:", font=("Arial", 14)).pack()
            self.message_length_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.message_length_entry.pack()

            self.write_var = tk.BooleanVar()
            tk.Checkbutton(self.additional_options_frame, text="Uložit dekódovanou zprávu jako textový soubor",
                           variable=self.write_var, font=("Arial", 14)).pack()

        elif selected_method == "DWT - BCH codes":
            tk.Label(self.additional_options_frame,
                     text="XOR Klíč (15 bitů):", font=("Arial", 14)).pack()
            self.xor_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.xor_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 1:", font=("Arial", 14)).pack()
            self.col_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.col_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Klíč 2:", font=("Arial", 14)).pack()
            self.row_key_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.row_key_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="Délka zprávy:", font=("Arial", 14)).pack()
            self.message_length_entry = tk.Entry(self.additional_options_frame, font=(
                "Arial", 14), width=50, validate="key", validatecommand=(validate_numeric, '%P'))
            self.message_length_entry.pack()

            tk.Label(self.additional_options_frame,
                     text="BCH Code:", font=("Arial", 14)).pack()
            self.bch_code_var = tk.StringVar()
            self.bch_code_combobox = ttk.Combobox(self.additional_options_frame, textvariable=self.bch_code_var, font=(
                "Arial", 14), values=[5, 7, 11], state='readonly')
            self.bch_code_combobox.pack()

            self.write_var = tk.BooleanVar()
            tk.Checkbutton(self.additional_options_frame, text="Uložit dekódovanou zprávu jako textový soubor",
                           variable=self.write_var, font=("Arial", 14)).pack()
        elif selected_method == "DCT psychovisual and object motion":
            # Frame for selecting motion blocks file
            motion_blocks_frame = tk.Frame(self.additional_options_frame)
            motion_blocks_frame.pack(fill=tk.X, pady=10)

            tk.Button(motion_blocks_frame, text="Vybrat soubor json",
                      command=self.select_motion_blocks_file, font=("Arial", 14)).pack(side=tk.RIGHT)
            self.motion_blocks_path_var = tk.StringVar()
            tk.Entry(motion_blocks_frame, textvariable=self.motion_blocks_path_var, font=(
                "Arial", 12), width=100, state='readonly').pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            tk.Label(self.additional_options_frame,
                     text="Klíč:", font=("Arial", 14)).pack()
            self.dct_key_entry = tk.Entry(
                self.additional_options_frame, font=("Arial", 14), width=50)
            self.dct_key_entry.pack()

            # Add checkbox for saving the result as a text file
            self.write_var = tk.BooleanVar()
            tk.Checkbutton(self.additional_options_frame, text="Uložit dekódovanou zprávu jako textový soubor",
                           variable=self.write_var, font=("Arial", 14)).pack()

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def select_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path_var.set(file_path)

    def select_motion_blocks_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")])
        if file_path:
            self.motion_blocks_path_var.set(file_path)

    def is_numeric(self, value_if_allowed):
        if value_if_allowed.isdigit() or value_if_allowed == "":
            return True
        else:
            return False

    def encode(self):
        method = self.selected_method.get()
        message = self.message_text.get("1.0", tk.END).strip()
        video_path = self.video_path_var.get()

        loading_window = LoadingWindow(self.master)

        def encode_task():
            if method == "LSB - Hamming code (7,4)":
                key1 = self.key1_entry.get()
                key2 = self.key2_entry.get()
                key3 = self.key3_entry.get()

                xor_random_key = np.random.randint(0, 2, size=7)

                try:
                    len_msg = hmc.hamming_encode(video_path, message, int(
                        key1), int(key2), int(key3), xor_random_key)
                    self.master.after(0, lambda: self.show_encode_result(
                        "LSB - Hamming code (7,4)", len_msg, key1, key2, key3, ''.join(map(str, xor_random_key))))

                except ValueError as e:
                    messagebox.showerror("Error", str(e))
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"An unexpected error occurred: {str(e)}")

            elif method == "DWT - BCH codes":
                bch_code = self.bch_code_var.get()
                col_key = self.col_key_entry.get()
                row_key = self.row_key_entry.get()

                xor_random_key = np.random.randint(0, 2, size=15)

                try:
                    message_len = bch_dwt.encode_bch_dwt(video_path, message, xor_random_key, int(
                        col_key), int(row_key), bch_num=int(bch_code))
                    self.master.after(0, lambda: self.show_encode_result(
                        "DWT - BCH codes", ''.join(map(str, xor_random_key)), message_len, col_key, row_key))

                except ValueError as e:
                    messagebox.showerror("Error", str(e))
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"An unexpected error occurred: {str(e)}")

            elif method == "DCT psychovisual and object motion":
                motion_blocks_file = "motion_blocks.json"
                try:
                    key = dct_psy.encode_dtc_psyschovisual(
                        video_path, message, motion_blocks_file, False)
                    self.master.after(0, lambda: self.show_encode_result(
                        "DCT psychovisual and object motion", key))
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"An unexpected error occurred: {str(e)}")

            loading_window.close()

        threading.Thread(target=encode_task, daemon=True).start()

    def show_encode_result(self, method, *args):
        if method == "LSB - Hamming code (7,4)":
            len_msg, key1, key2, key3, xor_key = args
            messagebox.showinfo(
                "Hotovo", f"Kódování dokončeno\nDélka zprávy: {len_msg} (slouží jako klíč k dekódování)\nklíč1: {int(key1)}\nklíč2: {int(key2)}\nklíč3: {int(key3)}\nXOR klíč: {xor_key}")
        elif method == "DWT - BCH codes":
            xor_key, message_len, col_key, row_key = args
            messagebox.showinfo(
                "Hotovo", f"Kódování dokončeno\nXor: {xor_key}\nKlíč 1: {col_key}\nKlíč 2: {row_key}\nDélka zprávy: {message_len}\n(slouží jako klíče k dekódování)")
        elif method == "DCT psychovisual and object motion":
            key = args[0]
            messagebox.showinfo(
                "Hotovo", f"Kódování dokončeno\nKlíč: {key}\n(slouží jako klíč k dekódování)")

        self.main_menu()

    def decode(self):
        method = self.selected_method.get()
        video_path = self.video_path_var.get()

        loading_window = LoadingWindow(self.master)

        def decode_task():
            if method == "LSB - Hamming code (7,4)":
                xor_key = self.xor_key_entry.get()
                key1 = self.key1_entry.get()
                key2 = self.key2_entry.get()
                key3 = self.key3_entry.get()
                message_length = self.message_length_entry.get()

                write_val = self.write_var.get()

                if len(xor_key) != 7 or not all(bit in '01' for bit in xor_key):
                    messagebox.showerror(
                        "Chyba", "XOR klíč musí být 7bitové binární číslo (pouze 0 a 1)")
                    loading_window.close()
                    return

                xor_key_arr = np.array([int(bit) for bit in xor_key])

                decoded_message = hmc.hamming_decode(video_path, int(key1), int(key2), int(
                    key3), int(message_length), xor_key_arr, write_file=write_val)
                self.master.after(
                    0, lambda: self.show_decode_result(decoded_message))

            elif method == "DWT - BCH codes":
                xor_key = self.xor_key_entry.get()
                col_key = self.col_key_entry.get()
                row_key = self.row_key_entry.get()
                message_length = self.message_length_entry.get()
                bch_code = self.bch_code_var.get()
                write_val = self.write_var.get()

                if len(xor_key) != 15 or not all(bit in '01' for bit in xor_key):
                    messagebox.showerror(
                        "Chyba", "XOR klíč musí být 15bitové binární číslo (pouze 0 a 1)")
                    loading_window.close()
                    return

                xor_key_arr = np.array([int(bit) for bit in xor_key])

                decoded_message = bch_dwt.decode_bch_dwt(video_path, int(message_length), xor_key_arr, int(
                    col_key), int(row_key), bch_num=int(bch_code), write_file=write_val)

                self.master.after(
                    0, lambda: self.show_decode_result(decoded_message))

            elif method == "DCT psychovisual and object motion":
                motion_blocks_file = self.motion_blocks_path_var.get()
                key = self.dct_key_entry.get()
                write_val = self.write_var.get()

                try:
                    decoded_message = dct_psy.decode_dtc_psyschovisual(
                        video_path, int(key), motion_blocks_file, write_val)
                    self.master.after(
                        0, lambda: self.show_decode_result(decoded_message))
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"An unexpected error occurred: {str(e)}")

            loading_window.close()

        threading.Thread(target=decode_task, daemon=True).start()

    def show_decode_result(self, decoded_message):
        messagebox.showinfo("Hotovo", f"Tajná zpráva:\n{decoded_message}")
        self.main_menu()


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TMenubutton', font=('Arial', 14))
    app = SteganographyApp(root)
    root.mainloop()
