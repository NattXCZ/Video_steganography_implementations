
# Video Steganography Implementations

Tato aplikace implementuje různé steganografické metody pro skrývání dat ve video souborech:

1. **LSB metoda pomocí Hammingova kódu (7, 4)**:
   Tato metoda využívá techniku Least Significant Bit (LSB) pro vložení dat do nejméně významných bitů pixelů. Zároveň využívá Hammingův kód (7, 4) pro detekci a opravu chyb, což zvyšuje odolnost proti poškození dat. Je vhodná pro situace, kdy je prioritou zachování kvality videa a přesnost extrakce skrytých dat.

2. **DWT metoda pomocí BCH kódů (15, 11)**:
   Tato metoda kombinuje diskrétní vlnkovou transformaci (DWT) pro rozklad obrazu na frekvenční pásma a BCH kódy pro robustní zakódování dat. Skrytá data jsou vložena do frekvenčních složek videa, které jsou méně nápadné pro lidské oko. Tato metoda je lepší pro situace, kde je důležitější odolnost vůči detekci než přesnost extrakce.

3. **Metoda založená na psychovizuální analýze DCT a pohybu objektů**:
   Tato metoda využívá diskrétní kosinovou transformaci (DCT) a analyzuje pohyb objektů ve videu. Tajná data jsou skryta v DCT koeficientech v oblastech videa, kde dochází k pohybu, což pomáhá minimalizovat viditelnost změn. Tato metoda je ideální pro videa s dostatečným pohybem, kde je klíčová maximální kvalita výsledného videa.

## Struktura projektu

- `data_testing/`: Obsahuje testovací video soubory
- `src/`: Zdrojový kód aplikace
  - `methods/`: Implementace jednotlivých steganografických metod
  - `utils/`: Pomocné nástroje pro zpracování videa a binárních dat
- `tests/`: Adresář s příkladový souborem s ukázkou výpočtů metrik pro testování.
- `main.py`: Hlavní spouštěcí skript aplikace
- `setup.py`: Skript pro instalaci aplikace




## Instalace a spuštění

### Požadavky
- Python 3.9 nebo novější
- FFmpeg

### Instalace FFmpeg
Pro běh programu je nutné mít nainstalován [FFmpeg](https://ffmpeg.org/). Doporučuji ke stahování využít [Chocolatey](https://chocolatey.org/install).

Postup instalace FFmpeg:
1. Otevřete příkazový řádek jako správce
2. Zadejte příkaz: `choco install ffmpeg-full`


### Instalace aplikace

1. Stáhněte si celou složku na své zařízení.

2. Přejděte do složky `video_steganography_implementations`:
   ```
   cd video_steganography_implementations
   ```

3. Nainstalujte aplikaci a všechny potřebné závislosti pomocí `setup.py`:
   ```
   pip install .
   ```



### Použití

Po instalaci můžete aplikaci spustit pomocí příkazu:
```
python main.py
```



## Podrobný návod práce s programem

Po spuštění se zobrazí úvodní menu obsahující stručný popis metod a dvě tlačítka: **Zakódovat** pro vložení zprávy do videa a **Dekódovat** pro dekódování zprávy ze stego videa.

### Vkládání zprávy

1. Vyberte metodu steganografie z rozbalovací nabídky v horní části okna.
2. Klikněte na tlačítko **Vybrat video** a zvolte video soubor (podporované formáty: MP4, AVI, MOV).
3. Do textového pole napište zprávu, kterou chcete skrýt. Zpráva může obsahovat pouze základní znaky bez diakritiky (např. písmena A-Z, číslice a běžné symboly).
4. Vyplňte požadované klíče a parametry podle zvolené metody. Doporučuje se použít klíče v rozmezí od 0 do 20. U metody využívající BCH codes je výhodné zadat z roletky hodnotu 5. Zachová zprávu co nejvíce původní a dostraní větší množství chyb.
5. Klikněte na **Potvrdit** pro zahájení vkládání zprávy.

Po vložení zprávy se stego video automaticky uloží jako `video.avi` do složky s programem. Po úspěšném vložení zprávy se zobrazí oznamovací okno s klíči, které jsou nezbytné pro dekódování zprávy z videa. Je důležité tyto klíče bezpečně zaznamenat, protože jejich ztráta znemožní dekódování. Aplikace také automaticky generuje náhodné klíče pro dodatečné zabezpečení.

### Dekódování zprávy

1. Vyberte stejnou metodu steganografie jako při vkládání.
2. Klikněte na tlačítko **Vybrat stego-video** a zvolte video se skrytou zprávou.
3. Zadejte stejné klíče a parametry, které byly použity při vkládání. V případě metody DCT psychovisual and object motion vložte odpovídající JSON soubor.
4. Klikněte na **Potvrdit** pro zahájení dekódování zprávy.

Pro úspěšné dekódování je naprosto nezbytné použít stejnou metodu a klíče jako při vkládání zprávy. Jakákoliv odchylka může vést k neúspěšnému dekódování nebo získání nesprávné zprávy.


## Řešení případné chybové hlášky

Pokud se objeví chybová hláška `OSError: Could not load shared object file: llvmlite.dll`, která se může objevit při importu knihovny `galois`. Tato knihovna následně používá `numba`, což vede k problému s `llvmlite`. Pro vyřešení problému postupujte následovně:

1. Zkuste nejprve přeinstalovat `numba` a `llvmlite`.
2. Pokud to nepomůže, stáhněte a nainstalujte Visual C++ Redistributable:
   - **Pro x64:** [Stáhnout VC Redist x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - **Další verze a informace:** [Visual Studio 2017 Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170)
3. Po instalaci restartujte počítač.