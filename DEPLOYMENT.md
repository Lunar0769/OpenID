# Deployment Guide

## Deploy to Render

1. Push your code to GitHub.
2. In the [Render dashboard](https://dashboard.render.com), create a new **Web Service** and connect your repository.
3. Render will detect `render.yaml` and pre-fill the service configuration.
4. Set the secret environment variables in the Render dashboard (see below).
5. Click **Deploy**.

Alternatively, install the [Render CLI](https://render.com/docs/cli) and run:

```bash
render deploy
```

## Required Environment Variables

Set these in the Render dashboard under **Environment → Environment Variables**:

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (from Render managed DB or external) |
| `REDIS_URL` | Redis connection string (Redis Cloud or Render Redis) |
| `STRIPE_API_KEY` | Stripe live secret key (`sk_live_...`) |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret (`whsec_...`) |
| `STRIPE_STARTER_PRICE_ID` | Stripe Price ID for the Starter plan |
| `STRIPE_GROWTH_PRICE_ID` | Stripe Price ID for the Growth plan |
| `CHECKOUT_SUCCESS_URL` | Redirect URL after successful Stripe checkout |
| `CHECKOUT_CANCEL_URL` | Redirect URL after cancelled Stripe checkout |
| `PORTAL_RETURN_URL` | Return URL from the Stripe customer portal |

Copy `.env.example` to `.env` for local development and fill in real values.

## Set Up Managed PostgreSQL and Redis

**PostgreSQL (Render):**
1. In the Render dashboard, create a new **PostgreSQL** instance (Starter plan, PostgreSQL 15).
2. Copy the **Internal Database URL** and set it as `DATABASE_URL` in your web service environment.

**Redis (Redis Cloud):**
1. Sign up at [Redis Cloud](https://redis.com/try-free/) and create a free database (30 MB).
2. Copy the connection string and set it as `REDIS_URL`.

## Run Database Migrations

After deploying (or on first deploy), run:

```bash
alembic upgrade head
```

On Render you can run this as a one-off job via the **Shell** tab in your service dashboard, or add it as a pre-deploy command.

## Configure Stripe Webhook

1. Go to the [Stripe Webhooks dashboard](https://dashboard.stripe.com/webhooks).
2. Click **Add endpoint**.
3. Set the endpoint URL to:
   ```
   https://<your-domain>/webhooks/stripe
   ```
4. Subscribe to the following events:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `customer.subscription.deleted`
5. After saving, copy the **Signing secret** (`whsec_...`) and set it as `STRIPE_WEBHOOK_SECRET` in your Render environment variables.

## TLS / HTTPS

TLS is provided automatically by Render via Let's Encrypt. All custom domains get a free certificate. HSTS headers (`max-age=31536000; includeSubDomains`) are added by the application middleware, and HTTP requests are automatically redirected to HTTPS (HTTP 301).
