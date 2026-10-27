import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    }).compileComponents();
  });

  it('should create the app component', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render dashboard title', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('#dashboard-title')?.textContent).toContain('Clinical Diabetes Risk Prediction Dashboard');
  });

  it('should display CI/CD automated deployment badge', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('#live-cicd-status')?.textContent).toContain('CI/CD Automated Deployment Active');
  });

  it('should toggle model selection to logistic regression', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.selectedModelKey).toBe('random_forest');
    app.selectModel('logistic_regression');
    expect(app.selectedModelKey).toBe('logistic_regression');
  });

  it('should load healthy preset and evaluate low risk', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.loadPreset('healthy');
    expect(app.formData.glucose_fasting).toBe(88);
    expect(app.predictionResult).toBeTruthy();
    expect(app.predictionResult?.prediction).toBe(0);
  });
});
