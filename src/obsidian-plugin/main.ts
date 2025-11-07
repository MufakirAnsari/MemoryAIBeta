import { App, Plugin, PluginSettingTab, Setting, TFile, WorkspaceLeaf, ItemView, Notice } from 'obsidian';
import { MemoryAISettings, DEFAULT_SETTINGS } from './settings';
import { MemoryAISuggestionModal } from './ui/suggestionModal';
import { MemoryGraphView, MEMORY_GRAPH_VIEW_TYPE } from './ui/memoryGraphView';
import { MemoryAIStatusBar } from './ui/statusBar';
import { MemoryAPIService } from './services/memoryAPIService';
import { SuggestionEngine } from './services/suggestionEngine';
import { GraphRenderer } from './services/graphRenderer';
import { PrivacyManager } from './services/privacyManager';

export default class MemoryAIPlugin extends Plugin {
	settings: MemoryAISettings;
	apiService: MemoryAPIService;
	suggestionEngine: SuggestionEngine;
	graphRenderer: GraphRenderer;
	privacyManager: PrivacyManager;
	statusBar: MemoryAIStatusBar;

	async onload() {
		await this.loadSettings();

		// Initialize services
		this.apiService = new MemoryAPIService(this.settings);
		this.suggestionEngine = new SuggestionEngine(this.apiService, this.settings);
		this.graphRenderer = new GraphRenderer();
		this.privacyManager = new PrivacyManager(this.settings);
		this.statusBar = new MemoryAIStatusBar(this.addStatusBarItem(), this.settings);

		// Register view
		this.registerView(
			MEMORY_GRAPH_VIEW_TYPE,
			(leaf) => new MemoryGraphView(leaf, this)
		);

		// Add ribbon icon
		this.addRibbonIcon('brain', 'MemoryAI', () => {
			this.showMemoryGraph();
		});

		// Add commands
		this.addCommand({
			id: 'get-suggestion',
			name: 'Get Smart Suggestion',
			callback: () => {
				this.getSmartSuggestion();
			}
		});

		this.addCommand({
			id: 'search-memory',
			name: 'Search Memory Graph',
			callback: () => {
				this.searchMemoryGraph();
			}
		});

		this.addCommand({
			id: 'toggle-mode',
			name: 'Toggle Halo/Focus Mode',
			callback: () => {
				this.toggleMode();
			}
		});

		this.addCommand({
			id: 'show-memory-graph',
			name: 'Show Memory Graph',
			callback: () => {
				this.showMemoryGraph();
			}
		});

		this.addCommand({
			id: 'generate-note-summary',
			name: 'Generate Note Summary',
			callback: () => {
				this.generateNoteSummary();
			}
		});

		this.addCommand({
			id: 'find-related-notes',
			name: 'Find Related Notes',
			callback: () => {
				this.findRelatedNotes();
			}
		});

		// Add settings tab
		this.addSettingTab(new MemoryAISettingTab(this.app, this));

		// Register event handlers
		this.registerEvent(
			this.app.workspace.on('file-open', (file) => {
				this.onFileOpen(file);
			})
		);

		this.registerEvent(
			this.app.vault.on('modify', (file) => {
				this.onFileModify(file);
			})
		);

		// Initialize status bar
		this.statusBar.initialize();

		// Show welcome notice
		if (this.settings.showWelcomeNotice) {
			new Notice('🧠 MemoryAI Enterprise activated! Your AI companion is ready.', 5000);
			this.settings.showWelcomeNotice = false;
			await this.saveSettings();
		}
	}

	onunload() {
		// Cleanup
		this.statusBar.cleanup();
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
	}

	async getSmartSuggestion(): Promise<void> {
		try {
			const activeFile = this.app.workspace.getActiveFile();
			if (!activeFile) {
				new Notice('No active file found');
				return;
			}

			const fileContent = await this.app.vault.read(activeFile);
			const cursor = this.getCursorPosition();

			// Show loading
			this.statusBar.showLoading();

			const suggestions = await this.suggestionEngine.getSuggestions({
				filePath: activeFile.path,
				content: fileContent,
				cursorPosition: cursor,
				vaultContext: this.getVaultContext()
			});

			this.statusBar.hideLoading();

			if (suggestions.length > 0) {
				new MemoryAISuggestionModal(this.app, suggestions, (selected) => {
					this.applySuggestion(selected, activeFile);
				}).open();
			} else {
				new Notice('No suggestions available right now');
			}

		} catch (error) {
			console.error('Error getting suggestion:', error);
			new Notice('Failed to get suggestion. Check your connection.');
			this.statusBar.hideLoading();
		}
	}

	async searchMemoryGraph(): Promise<void> {
		const query = await this.showInputDialog({
			title: 'Search Memory Graph',
			placeholder: 'What would you like to find?',
			prompt: 'Search through your personal knowledge graph'
		});

		if (!query) return;

		try {
			const results = await this.apiService.searchMemory({
				userId: this.settings.userId,
				query: query,
				limit: 10
			});

			if (results.length > 0) {
				this.showSearchResults(results);
			} else {
				new Notice('No memories found matching your query');
			}
		} catch (error) {
			console.error('Error searching memory:', error);
			new Notice('Failed to search memory graph');
		}
	}

	toggleMode(): void {
		this.settings.isHaloMode = !this.settings.isHaloMode;
		this.saveSettings();
		
		const mode = this.settings.isHaloMode ? 'Halo' : 'Focus';
		new Notice(`Switched to ${mode} mode`);
		
		this.statusBar.updateMode(this.settings.isHaloMode);
	}

	async showMemoryGraph(): Promise<void> {
		const { workspace } = this.app;
		
		let leaf: WorkspaceLeaf | null = null;
		const leaves = workspace.getLeavesOfType(MEMORY_GRAPH_VIEW_TYPE);
		
		if (leaves.length > 0) {
			leaf = leaves[0];
		} else {
			leaf = workspace.getRightLeaf(false);
			await leaf.setViewState({
				type: MEMORY_GRAPH_VIEW_TYPE,
				active: true
			});
		}
		
		workspace.revealLeaf(leaf);
	}

	async generateNoteSummary(): Promise<void> {
		const activeFile = this.app.workspace.getActiveFile();
		if (!activeFile) {
			new Notice('No active file found');
			return;
		}

		try {
			const fileContent = await this.app.vault.read(activeFile);
			
			const summary = await this.apiService.generateSummary({
				userId: this.settings.userId,
				content: fileContent,
				filePath: activeFile.path
			});

			// Add summary to the beginning of the note
			const summaryContent = `## Summary\n\n${summary}\n\n---\n\n`;
			const updatedContent = summaryContent + fileContent;
			
			await this.app.vault.modify(activeFile, updatedContent);
			new Notice('Summary generated and added to note');

		} catch (error) {
			console.error('Error generating summary:', error);
			new Notice('Failed to generate summary');
		}
	}

	async findRelatedNotes(): Promise<void> {
		const activeFile = this.app.workspace.getActiveFile();
		if (!activeFile) {
			new Notice('No active file found');
			return;
		}

		try {
			const fileContent = await this.app.vault.read(activeFile);
			
			const relatedNotes = await this.apiService.findRelated({
				userId: this.settings.userId,
				content: fileContent,
				filePath: activeFile.path,
				limit: 5
			});

			if (relatedNotes.length > 0) {
				this.showRelatedNotes(relatedNotes);
			} else {
				new Notice('No related notes found');
			}

		} catch (error) {
			console.error('Error finding related notes:', error);
			new Notice('Failed to find related notes');
		}
	}

	private getCursorPosition(): number {
		// Get current cursor position in the active editor
		const activeView = this.app.workspace.getActiveViewOfType(ItemView);
		if (activeView && 'editor' in activeView) {
			// This would need to be adapted based on the actual editor interface
			return 0;
		}
		return 0;
	}

	private getVaultContext(): any {
		const files = this.app.vault.getMarkdownFiles();
		return {
			totalFiles: files.length,
			recentFiles: files.slice(0, 10).map(f => f.path)
		};
	}

	private async showInputDialog(options: {
		title: string;
		placeholder?: string;
		prompt?: string;
	}): Promise<string | undefined> {
		// Implementation would show a modal dialog
		return new Promise((resolve) => {
			// Mock implementation
			resolve('sample query');
		});
	}

	private showSearchResults(results: any[]): void {
		// Implementation would show search results in a modal or sidebar
		console.log('Search results:', results);
	}

	private showRelatedNotes(notes: any[]): void {
		// Implementation would show related notes
		console.log('Related notes:', notes);
	}

	private async applySuggestion(suggestion: any, file: TFile): Promise<void> {
		try {
			const fileContent = await this.app.vault.read(file);
			
			// Apply suggestion based on type
			let updatedContent = fileContent;
			
			switch (suggestion.type) {
				case 'memory-based':
					// Add memory-related content
					updatedContent += `\n\n## Related Memory\n${suggestion.content}`;
					break;
				case 'pattern-based':
					// Apply pattern suggestion
					updatedContent = this.applyPatternSuggestion(fileContent, suggestion);
					break;
				case 'contextual':
					// Add contextual information
					updatedContent = this.applyContextualSuggestion(fileContent, suggestion);
					break;
				default:
					// Insert at cursor position
					updatedContent = this.insertAtCursor(fileContent, suggestion.content);
			}

			await this.app.vault.modify(file, updatedContent);
			new Notice('Suggestion applied successfully');

			// Submit feedback
			await this.apiService.submitFeedback({
				suggestionId: suggestion.id,
				accepted: true,
				userId: this.settings.userId
			});

		} catch (error) {
			console.error('Error applying suggestion:', error);
			new Notice('Failed to apply suggestion');
		}
	}

	private applyPatternSuggestion(content: string, suggestion: any): string {
		// Implementation would apply pattern-based suggestions
		return content + `\n\n${suggestion.content}`;
	}

	private applyContextualSuggestion(content: string, suggestion: any): string {
		// Implementation would apply contextual suggestions
		return content + `\n\n> ${suggestion.content}`;
	}

	private insertAtCursor(content: string, suggestion: string): string {
		// Implementation would insert at cursor position
		return content + suggestion;
	}

	private onFileOpen(file: TFile | null): void {
		if (file) {
			console.log('File opened:', file.path);
			// Could trigger contextual suggestions
		}
	}

	private onFileModify(file: TFile): void {
		console.log('File modified:', file.path);
		// Could trigger pattern analysis
	}
}

class MemoryAISettingTab extends PluginSettingTab {
	plugin: MemoryAIPlugin;

	constructor(app: App, plugin: MemoryAIPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;

		containerEl.empty();

		containerEl.createEl('h2', { text: 'MemoryAI Enterprise Settings' });

		// API Settings
		containerEl.createEl('h3', { text: 'API Configuration' });

		new Setting(containerEl)
			.setName('API URL')
			.setDesc('MemoryAI API endpoint')
			.addText(text => text
				.setPlaceholder('http://localhost:8082')
				.setValue(this.plugin.settings.apiUrl)
				.onChange(async (value) => {
					this.plugin.settings.apiUrl = value;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('User ID')
			.setDesc('Your unique user identifier')
			.addText(text => text
				.setPlaceholder('obsidian_user')
				.setValue(this.plugin.settings.userId)
				.onChange(async (value) => {
					this.plugin.settings.userId = value;
					await this.plugin.saveSettings();
				}));

		// Privacy Settings
		containerEl.createEl('h3', { text: 'Privacy Settings' });

		new Setting(containerEl)
			.setName('Privacy Level')
			.setDesc('Choose how your data is processed')
			.addDropdown(dropdown => dropdown
				.addOption('local', 'Local Only (Most Private)')
				.addOption('hybrid', 'Hybrid (Balanced)')
				.addOption('cloud', 'Cloud (Most Capable)')
				.setValue(this.plugin.settings.privacyLevel)
				.onChange(async (value: 'local' | 'hybrid' | 'cloud') => {
					this.plugin.settings.privacyLevel = value;
					await this.plugin.saveSettings();
				}));

		// Suggestion Settings
		containerEl.createEl('h3', { text: 'Suggestion Settings' });

		new Setting(containerEl)
			.setName('Auto Suggest')
			.setDesc('Automatically show suggestions while typing')
			.addToggle(toggle => toggle
				.setValue(this.plugin.settings.autoSuggest)
				.onChange(async (value) => {
					this.plugin.settings.autoSuggest = value;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Suggestion Timeout')
			.setDesc('Timeout for suggestion requests (milliseconds)')
			.addText(text => text
				.setPlaceholder('2000')
				.setValue(this.plugin.settings.suggestionTimeout.toString())
				.onChange(async (value) => {
					const timeout = parseInt(value);
					if (!isNaN(timeout)) {
						this.plugin.settings.suggestionTimeout = timeout;
						await this.plugin.saveSettings();
					}
				}));

		// Display Settings
		containerEl.createEl('h3', { text: 'Display Settings' });

		new Setting(containerEl)
			.setName('Default Mode')
			.setDesc('Default interaction mode')
			.addDropdown(dropdown => dropdown
				.addOption('halo', 'Halo Mode')
				.addOption('focus', 'Focus Mode')
				.setValue(this.plugin.settings.isHaloMode ? 'halo' : 'focus')
				.onChange(async (value: 'halo' | 'focus') => {
					this.plugin.settings.isHaloMode = value === 'halo';
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Show Confidence Scores')
			.setDesc('Display confidence scores for suggestions')
			.addToggle(toggle => toggle
				.setValue(this.plugin.settings.showConfidence)
				.onChange(async (value) => {
					this.plugin.settings.showConfidence = value;
					await this.plugin.saveSettings();
				}));
	}
}