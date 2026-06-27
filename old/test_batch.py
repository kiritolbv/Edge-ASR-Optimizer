from pathlib import Path
from processor import AudioMLOptimizer, PipelineConfig
import soundfile as sf

optimizer = AudioMLOptimizer(PipelineConfig(denoise_model_path=None))

fichiers = list(Path('input').glob('*.wav'))

print(f"Traitement de {len(fichiers)} fichiers...")

for chemin in fichiers:
    print(f"\nTraitement : {chemin.name}")
    try:
        audio_nettoyee = optimizer.process_file(chemin)
        
        # Sauvegarde
        output_path = Path('output') / f"{chemin.stem}_denoised.wav"
        output_path.parent.mkdir(exist_ok=True)
        sf.write(output_path, audio_nettoyee, 16000)
        print(f"  ✅ Sauvegardé : {output_path}")
        
    except Exception as e:
        print(f"  ❌ Erreur : {e}")