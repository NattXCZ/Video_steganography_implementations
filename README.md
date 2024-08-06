
# Video Steganography Implementations

Tato aplikace implementuje různé steganografické metody pro skrývání dat ve video souborech:

1. **LSB metoda pomocí Hammingova kódu (7, 4)**:
   Tato metoda využívá techniku Least Significant Bit (LSB) pro vložení dat do nejméně významných bitů pixelů. Zároveň využívá Hammingův kód (7, 4) pro detekci a opravu chyb, což zvyšuje odolnost proti poškození dat. Je vhodná pro situace, kdy je prioritou zachování kvality videa a přesnost extrakce skrytých dat.

2. **DWT metoda pomocí BCH kódů (15, 11)**:
   Tato metoda kombinuje diskrétní vlnkovou transformaci (DWT) pro rozklad obrazu na frekvenční pásma a BCH kódy pro robustní zakódování dat. Skrytá data jsou vložena do frekvenčních složek videa, které jsou méně nápadné pro lidské oko. Tato metoda je lepší pro situace, kde je důležitější odolnost vůči detekci než přesnost extrakce.

3. **Metoda založená na psychovizuální analýze DCT a pohybu objektu**:
   Tato metoda využívá diskrétní kosinovou transformaci (DCT) a analyzuje pohyb objektů ve videu. Tajná data jsou skryta v DCT koeficientech v oblastech videa, kde dochází k pohybu, což pomáhá minimalizovat viditelnost změn. Tato metoda je ideální pro videa s dostatečným pohybem, kde je klíčová maximální kvalita výsledného videa.

## Struktura projektu

- `data_testing/`: Obsahuje testovací video soubory
- `src/`: Zdrojový kód aplikace
  - `methods/`: Implementace jednotlivých steganografických metod
  - `utils/`: Pomocné nástroje pro zpracování videa a binárních dat
- `main.py`: Hlavní spouštěcí skript aplikace
- `setup.py`: Skript pro instalaci aplikace

## Instalace

1. Ujistěte se, že máte nainstalovaný Python 3.9 nebo novější verzi.

2. Naklonujte tento repozitář:
   ```
   git clone https://github.com/NattXCZ/Video_steganography_implementations.git
   ```

3. Přejděte do složky `video_steganography_implementations`:
   ```
   cd video_steganography_implementations
   ```

4. Nainstalujte aplikaci a všechny potřebné závislosti pomocí `setup.py`:
   ```
   pip install .
   ```

## Použití

Po instalaci můžete aplikaci spustit pomocí příkazu:
```
python main.py
```

### Podrobný návod

Po spuštění se zobrazí úvodní menu obsahující stručný popis metod a dvě tlačítka: **Zakódovat** pro vložení zprávy do videa a **Dekódovat** pro dekódování zprávy ze stego videa.

#### Vkládání zprávy

1. Vyberte metodu steganografie z rozbalovací nabídky v horní části okna.
2. Klikněte na tlačítko **Vybrat video** a zvolte video soubor (podporované formáty: MP4, AVI, MOV).
3. Do textového pole napište zprávu, kterou chcete skrýt. Zpráva může obsahovat pouze základní znaky bez diakritiky (např. písmena A-Z, číslice a běžné symboly).
4. Vyplňte požadované klíče a parametry podle zvolené metody. Doporučuje se použít klíče v rozmezí od 0 do 20.
5. Klikněte na **Potvrdit** pro zahájení vkládání zprávy.

Po vložení zprávy se stego video automaticky uloží jako `video.avi` do složky s programem. Po úspěšném vložení zprávy se zobrazí oznamovací okno s klíči, které jsou nezbytné pro dekódování zprávy z videa. Je důležité tyto klíče bezpečně zaznamenat, protože jejich ztráta znemožní dekódování. Aplikace může také automaticky generovat náhodné klíče pro dodatečné zabezpečení.

#### Dekódování zprávy

1. Vyberte stejnou metodu steganografie jako při vkládání.
2. Klikněte na tlačítko **Vybrat stego-video** a zvolte video se skrytou zprávou.
3. Zadejte stejné klíče a parametry, které byly použity při vkládání. V případě metody DCT psychovisual and object motion vložte odpovídající JSON soubor.
4. Klikněte na **Potvrdit** pro zahájení dekódování zprávy.

Pro úspěšné dekódování je naprosto nezbytné použít stejnou metodu a klíče jako při vkládání zprávy. Jakákoliv odchylka může vést k neúspěšnému dekódování nebo získání nesprávné zprávy.
