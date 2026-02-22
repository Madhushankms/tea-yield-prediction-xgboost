# Tea Yield Prediction - Frontend

Modern Next.js 14 frontend with TypeScript and Tailwind CSS for tea yield prediction system.

## 📁 Directory Structure

```
frontend/
├── app/
│   ├── layout.tsx       # Root layout
│   ├── page.tsx         # Main page
│   └── globals.css      # Global styles
├── components/
│   ├── PredictionForm.tsx   # Input form
│   ├── ResultCard.tsx       # Results display
│   └── FeatureChart.tsx     # Feature importance chart
├── lib/
│   └── api.ts               # API client utilities
├── public/                  # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── next.config.js
└── README.md
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
npm install
# or
yarn install
```

### 2. Configure Environment (Optional)

Create `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Default is `http://localhost:8000` if not specified.

### 3. Start Development Server

```bash
npm run dev
# or
yarn dev
```

Frontend will be available at: http://localhost:3000

### 4. Build for Production

```bash
npm run build
npm start
```

## 🎨 Components

### PredictionForm.tsx

Input form for tea farm parameters.

**Features:**

- 9 input fields with validation
- Visual icons for each parameter
- Real-time value updates
- Field descriptions
- Error handling
- Responsive design

**Props:**

```typescript
interface PredictionFormProps {
  onPredictionComplete: (result: PredictionResult) => void;
  onPredictionStart: () => void;
}
```

### ResultCard.tsx

Display prediction results with insights.

**Features:**

- Large prediction display
- Yield category classification
- Top 3 influential features
- Feature importance chart integration
- Actionable recommendations
- Color-coded yield categories

**Props:**

```typescript
interface ResultCardProps {
  result: PredictionResult;
}
```

### FeatureChart.tsx

Interactive bar chart for feature importance.

**Features:**

- Horizontal bar chart (recharts)
- Color-coded by importance
- Custom tooltips
- Responsive sizing
- Legend with impact levels

**Props:**

```typescript
interface FeatureChartProps {
  feature_importance: {
    [key: string]: number;
  };
}
```

## 🔌 API Integration

### API Client (lib/api.ts)

```typescript
// Health check
const status = await checkHealth();

// Make prediction
const result = await predictYield({
  rainfall: 2500,
  temperature: 24,
  fertilizer: 500,
  soil_ph: 5.0,
  humidity: 80,
  altitude: 1200,
  sunlight_hours: 6,
  plant_age: 20,
  pruning_frequency: 3,
});

// Get features list
const features = await getFeatures();
```

### Type Definitions

```typescript
interface PredictionInput {
  rainfall: number;
  temperature: number;
  fertilizer: number;
  soil_ph: number;
  humidity: number;
  altitude: number;
  sunlight_hours: number;
  plant_age: number;
  pruning_frequency: number;
}

interface PredictionResult {
  prediction: number;
  feature_importance: {
    [key: string]: number;
  };
}
```

## 🎯 Features

### User Experience

- ✅ Clean, modern UI with gradients
- ✅ Loading spinner during API calls
- ✅ Error messages with retry options
- ✅ Responsive design (mobile + desktop)
- ✅ Visual feedback on interactions
- ✅ Accessibility considerations

### Data Visualization

- ✅ Feature importance bar chart
- ✅ Color-coded yield categories
- ✅ Top influential factors display
- ✅ Interactive tooltips
- ✅ Legend and labels

### Form Validation

- ✅ Required field validation
- ✅ Min/max value constraints
- ✅ Real-time error feedback
- ✅ Type-safe inputs
- ✅ Clear field descriptions

## 🎨 Styling

### Tailwind CSS

Custom theme with green color palette:

```typescript
colors: {
  primary: {
    50: '#f0fdf4',   // Very light green
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',  // Main brand color
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',  // Dark green
  }
}
```

### Global Styles

- Custom scrollbar styling
- Gradient backgrounds
- Smooth transitions
- Consistent spacing

## 📱 Responsive Design

### Breakpoints

- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

### Grid Layout

```tsx
// Two-column on desktop, single column on mobile
<div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
  <PredictionForm />
  <ResultCard />
</div>
```

## 🧪 Testing

### Manual Testing

1. **Form Validation**
   - Enter invalid values
   - Submit empty form
   - Test min/max limits

2. **API Integration**
   - Successful prediction
   - API error handling
   - Network timeout

3. **UI Responsiveness**
   - Mobile view (375px)
   - Tablet view (768px)
   - Desktop view (1920px)

### Test Scenarios

```typescript
// Test prediction flow
1. Open http://localhost:3000
2. Fill form with default values
3. Click "Predict Tea Yield"
4. Verify loading spinner appears
5. Verify result displays correctly
6. Check feature importance chart
```

## 🛠️ Development

### Add New Input Field

1. Update `PredictionInput` type in `lib/api.ts`
2. Add field to `inputFields` array in `PredictionForm.tsx`
3. Update backend schema accordingly

### Customize Theme

Edit `tailwind.config.ts`:

```typescript
theme: {
  extend: {
    colors: {
      // Add custom colors
    },
    fontFamily: {
      // Add custom fonts
    }
  }
}
```

### Add New Component

```typescript
// components/MyComponent.tsx
interface MyComponentProps {
  // props
}

export default function MyComponent({ }: MyComponentProps) {
  return (
    // component JSX
  )
}
```

## 📦 Dependencies

### Core

- **next**: 14.0.4 - React framework
- **react**: 18.2.0 - UI library
- **react-dom**: 18.2.0 - React DOM

### Utilities

- **axios**: 1.6.2 - HTTP client
- **recharts**: 2.10.3 - Chart library

### TypeScript

- **typescript**: 5.3.3
- **@types/node**: 20.10.5
- **@types/react**: 18.2.45
- **@types/react-dom**: 18.2.18

### Styling

- **tailwindcss**: 3.3.6
- **autoprefixer**: 10.4.16
- **postcss**: 8.4.32

### Development

- **eslint**: 8.56.0
- **eslint-config-next**: 14.0.4

## 🔧 Troubleshooting

### Port already in use

```bash
npm run dev -- -p 3001
```

### API connection failed

1. Ensure backend is running on port 8000
2. Check `NEXT_PUBLIC_API_URL` in `.env.local`
3. Verify CORS settings in backend

### Module not found

```bash
rm -rf node_modules package-lock.json
npm install
```

### Build errors

```bash
npm run build
# Check for TypeScript errors
```

### Chart not rendering

1. Ensure recharts is installed
2. Check browser console for errors
3. Verify data format matches expected type

## 🚀 Deployment

### Vercel (Recommended)

```bash
npm install -g vercel
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables

Set in production:

```env
NEXT_PUBLIC_API_URL=https://your-backend-api.com
```

## 🎯 Best Practices

### Code Quality

- ✅ TypeScript strict mode enabled
- ✅ ESLint configuration
- ✅ Functional components only
- ✅ Proper error handling
- ✅ Type-safe API calls

### Performance

- ✅ Next.js automatic code splitting
- ✅ Image optimization (Next.js Image)
- ✅ Lazy loading components
- ✅ Minimal bundle size

### User Experience

- ✅ Loading states for async operations
- ✅ Error recovery mechanisms
- ✅ Clear user feedback
- ✅ Accessible UI elements
- ✅ Responsive design

## 📚 Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com)
- [Recharts](https://recharts.org)
- [TypeScript](https://www.typescriptlang.org)

---

**Part of the Tea Yield Prediction Full-Stack System**
