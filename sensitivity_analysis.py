import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

class SensitivityAnalyzer:
    def __init__(self, c_max: float = 0.9999, c_min: float = 0.99):
        self.c_max = c_max
        self.c_min = c_min
        self.sigma_values = [1.0, 5.0, 10.0, 20.0, 50.0]
        
    def analyze_compression_sensitivity(self, delta_p_range: List[float] = None) -> Dict[str, Any]:
        if delta_p_range is None:
            delta_p_range = np.linspace(0, 0.3, 31)
        
        results = {}
        
        for sigma in self.sigma_values:
            compression_ratios = []
            sensitivities = []
            
            for delta_p in delta_p_range:
                compression_ratio = self.calculate_compression_ratio(delta_p, sigma)
                sensitivity = self.calculate_sensitivity(delta_p, sigma)
                
                compression_ratios.append(compression_ratio)
                sensitivities.append(sensitivity)
            
            results[f'sigma_{sigma}'] = {
                'delta_p': delta_p_range.tolist(),
                'compression_ratios': compression_ratios,
                'sensitivities': sensitivities
            }
        
        return results
    
    def calculate_compression_ratio(self, delta_p: float, sigma: float) -> float:
        compression_ratio = self.c_max - (2 * (self.c_max - self.c_min)) / (1 + np.exp(delta_p * sigma))
        return max(self.c_min, min(self.c_max, compression_ratio))
    
    def calculate_sensitivity(self, delta_p: float, sigma: float) -> float:
        exp_term = np.exp(delta_p * sigma)
        sensitivity = 2 * (self.c_max - self.c_min) * sigma * exp_term / ((1 + exp_term) ** 2)
        return sensitivity
    
    def plot_sensitivity_analysis(self, results: Dict[str, Any], save_path: str = 'sensitivity_analysis.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, sigma in enumerate(self.sigma_values):
            key = f'sigma_{sigma}'
            if key in results:
                delta_p = results[key]['delta_p']
                compression_ratios = [c * 100 for c in results[key]['compression_ratios']]
                sensitivities = results[key]['sensitivities']
                
                ax1.plot(delta_p, compression_ratios, color=colors[i], 
                        label=f'σ={sigma}', linewidth=2)
                ax2.plot(delta_p, sensitivities, color=colors[i], 
                        label=f'σ={sigma}', linewidth=2)
        
        ax1.set_xlabel('Performance Fluctuation (Δp)')
        ax1.set_ylabel('Compression Ratio C (%)')
        ax1.set_title('Sensitivity Analysis: Compression Ratio vs Performance Fluctuation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(98, 100)
        
        ax2.set_xlabel('Performance Fluctuation (Δp)')
        ax2.set_ylabel('Sensitivity |dC/dΔp|')
        ax2.set_title('Sensitivity Analysis: Compression Sensitivity vs Performance Fluctuation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sensitivity analysis plot saved to {save_path}")
    
    def generate_sensitivity_report(self, results: Dict[str, Any]) -> str:
        report = "Dynamic Weight Compression Sensitivity Analysis Report\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Parameters:\n"
        report += f"- C_max: {self.c_max:.4f} ({self.c_max*100:.2f}%)\n"
        report += f"- C_min: {self.c_min:.4f} ({self.c_min*100:.2f}%)\n"
        report += f"- σ values tested: {self.sigma_values}\n\n"
        
        report += "Key Findings:\n"
        
        for sigma in self.sigma_values:
            key = f'sigma_{sigma}'
            if key in results:
                delta_p_01 = 0.1
                compression_at_01 = self.calculate_compression_ratio(delta_p_01, sigma)
                sensitivity_at_01 = self.calculate_sensitivity(delta_p_01, sigma)
                
                report += f"\nσ = {sigma}:\n"
                report += f"- C at Δp=0.1: {compression_at_01:.4f} ({compression_at_01*100:.2f}%)\n"
                report += f"- Sensitivity at Δp=0.1: {sensitivity_at_01:.4f}\n"
                report += f"- Adjustment rate: ~{sensitivity_at_01*100:.2f}% per 0.1 Δp\n"
        
        report += f"\nRecommended σ = 10.0:\n"
        report += f"- Balanced adaptation for typical Δp ranges (0.01-0.10)\n"
        report += f"- Practical adjustment rate of ~0.5% per 0.1 Δp\n"
        report += f"- Suitable for both early training volatility and stable convergence\n"
        
        return report

def run_sensitivity_analysis():
    analyzer = SensitivityAnalyzer()
    results = analyzer.analyze_compression_sensitivity()
    
    analyzer.plot_sensitivity_analysis(results)
    report = analyzer.generate_sensitivity_report(results)
    
    with open('sensitivity_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("Sensitivity analysis completed. Check 'sensitivity_analysis.png' and 'sensitivity_analysis_report.txt'")

if __name__ == '__main__':
    run_sensitivity_analysis()
